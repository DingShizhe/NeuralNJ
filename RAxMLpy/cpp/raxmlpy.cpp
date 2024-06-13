#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <iostream>
#include <utility>

#include <algorithm>
#include <chrono>
#include <cstdint>

#include <cstdarg>
#include <search.h>
#include <cstdio>
#include <ctime>
#include <cstring>


#include <memory>

#include "version.h"
#include "common.h"
#include "MSA.hpp"
#include "Options.hpp"
#include "CommandLineParser.hpp"
#include "Optimizer.hpp"
#include "PartitionInfo.hpp"
#include "PartitionedMSAView.hpp"
#include "ParsimonyMSA.hpp"
#include "TreeInfo.hpp"
#include "io/file_io.hpp"
#include "io/binary_io.hpp"
#include "ParallelContext.hpp"
#include "loadbalance/LoadBalancer.hpp"
#include "loadbalance/CoarseLoadBalancer.hpp"
#include "bootstrap/BootstrapGenerator.hpp"
#include "bootstrap/BootstopCheck.hpp"
#include "bootstrap/TransferBootstrapTree.hpp"
#include "bootstrap/ConsensusTree.hpp"
#include "autotune/ResourceEstimator.hpp"
#include "ICScoreCalculator.hpp"
#include "topology/RFDistCalculator.hpp"
#include "topology/ConstraintTree.hpp"
#include "util/EnergyMonitor.hpp"

#ifdef _RAXML_TERRAPHAST
#include "terraces/TerraceWrapper.hpp"
#endif


extern "C" {

}


using namespace std;


enum class RaxmlRunPhase
{
  start,
  mlsearch,
  bootstrap,
  finish
};


struct RaxmlInstance
{
  Options opts;
  shared_ptr<PartitionedMSA> parted_msa;
  unique_ptr<ParsimonyMSA> parted_msa_parsimony;
  map<BranchSupportMetric, shared_ptr<SupportTree> > support_trees;
  shared_ptr<ConsensusTree> consens_tree;

  TreeList start_trees;
  BootstrapReplicateList bs_reps;
  TreeList bs_start_trees;

  intVector bs_seeds;

  /* IDs of the trees that have been already inferred (eg after resuming from a checkpoint) */
  IDSet done_ml_trees;
  IDSet done_bs_trees;

  // load balancing
  PartitionAssignmentList proc_part_assign;
  unique_ptr<LoadBalancer> load_balancer;
  unique_ptr<CoarseLoadBalancer> coarse_load_balancer;

  // bootstopping convergence test, only autoMRE is supported for now
  unique_ptr<BootstopCheckMRE> bootstop_checker;
  bool bs_converged;
  RaxmlRunPhase run_phase;
  double used_wh;

  // mapping taxon name -> tip_id/clv_id in the tree
  NameIdMap tip_id_map;

  // mapping tip_id in the tree (array index) -> sequence index in MSA
  IDVector tip_msa_idmap;

 // unique_ptr<TerraceWrapper> terrace_wrapper;

//  unique_ptr<RandomGenerator> starttree_seed_gen;
//  unique_ptr<RandomGenerator> bootstrap_seed_gen;

  unique_ptr<NewickStream> start_tree_stream;

  /* this is just a dummy random tree used for convenience, e,g, if we need tip labels or
   * just 'any' valid tree for the alignment at hand */
  Tree random_tree;

  /* topological constraint */
  Tree constraint_tree;

  MLTree ml_tree;

  unique_ptr<RFDistCalculator> dist_calculator;
  AncestralStatesSharedPtr ancestral_states;
  vector<vector<doubleVector>> persite_loglh;      // per-tree -> per-partition -> per-site

//   vector<RaxmlWorker> workers;
//   RaxmlWorker& get_worker() { return workers.at(ParallelContext::local_group_id()); }

  RaxmlInstance() : bs_converged(false), run_phase(RaxmlRunPhase::start), used_wh(0) {}
};


/* a callback function for performing a full traversal */
static int cb_full_traversal(pll_unode_t * node)
{
  PLLMOD_UNUSED(node);
  return PLL_SUCCESS;
}


/* a callback function for performing a partial traversal on invalid CLVs */
static int cb_partial_traversal(pll_unode_t * node)
{
  /* do not include tips */
  if (!node->next) return PLL_FAILURE;

  pllmod_treeinfo_t * treeinfo = (pllmod_treeinfo_t *) node->data;

  /* if clv is invalid, traverse the subtree to compute it */
  if (treeinfo->active_partition == PLLMOD_TREEINFO_PARTITION_ALL)
  {
    /* check if at least one per-partition CLV is invalid */
    for (unsigned int i = 0; i < treeinfo->init_partition_count; ++i)
    {
      unsigned int p = treeinfo->init_partition_idx[i];
      if (treeinfo->clv_valid[p][node->node_index] == 0)
        return PLL_SUCCESS;
    }

    /* CLVs for all partitions are valid -> skip subtree */
    return PLL_FAILURE;
  }
  else
    return (treeinfo->clv_valid[treeinfo->active_partition][node->node_index] == 0);
}


static double treeinfo_compute_loglh_dsz(pllmod_treeinfo_t * treeinfo,
                                     int incremental,
                                     int update_pmatrices,
                                     double ** persite_lnl)
{
  /* tree root must be an inner node! */
  assert(!pllmod_utree_is_tip(treeinfo->root));

  unsigned int traversal_size = 0;
  unsigned int ops_count;
  unsigned int i, p;

  const double LOGLH_NONE = (double) NAN;
  double total_loglh = 0.0;
  const int old_active_partition = treeinfo->active_partition;

  /* NOTE: in unlinked brlen mode, up-to-date brlens for partition p
   * have to be prefetched to treeinfo->branch_lengths[p] !!! */
  int collect_brlen =
      (treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED ? 0 : 1);

  pllmod_treeinfo_set_active_partition(treeinfo, PLLMOD_TREEINFO_PARTITION_ALL);

  /* we need full traversal in 2 cases: 1) update p-matrices, 2) update all CLVs */
  if (!incremental || (update_pmatrices && collect_brlen))
  {
    /* perform a FULL postorder traversal of the unrooted tree */
    if (!pll_utree_traverse(treeinfo->root,
                            PLL_TREE_TRAVERSE_POSTORDER,
                            cb_full_traversal,
                            treeinfo->travbuffer,
                            &traversal_size))
      return LOGLH_NONE;
  }

  /* update p-matrices if asked for */
  if (update_pmatrices)
  {
    if (collect_brlen)
    {
      assert(traversal_size == treeinfo->tip_count * 2 - 2);
      for (i = 0; i < traversal_size; ++i)
       {
         pll_unode_t * node = treeinfo->travbuffer[i];
         treeinfo->branch_lengths[0][node->pmatrix_index] = node->length;
       }
    }

    pllmod_treeinfo_update_prob_matrices(treeinfo, !incremental);
  }

  if (incremental)
  {
    /* compute partial traversal and update only invalid CLVs */
    if (!pll_utree_traverse(treeinfo->root,
                            PLL_TREE_TRAVERSE_POSTORDER,
                            cb_partial_traversal,
                            treeinfo->travbuffer,
                            &traversal_size))
      return LOGLH_NONE;
  }

  /* create operations based on partial traversal obtained above */
  pll_utree_create_operations(treeinfo->travbuffer,
                              traversal_size,
                              NULL,
                              NULL,
                              treeinfo->operations,
                              NULL,
                              &ops_count);

  treeinfo->counter += ops_count;

//  printf("Traversal size (%s): %u\n", incremental ? "part" : "full", ops_count);

  /* iterate over all partitions (we assume that traversal is the same) */

  for (p = 0; p < treeinfo->partition_count; ++p)
  {
    if (!treeinfo->partitions[p])
    {
      /* this partition will be computed by another thread(s) */
      treeinfo->partition_loglh[p] = 0.0;
    //   std::cout << "Warning! !treeinfo->partitions[p]" << std::endl;
      continue;
    }

    /* all subsequent operation will affect current partition only */
    pllmod_treeinfo_set_active_partition(treeinfo, (int)p);

    /* use the operations array to compute all ops_count inner CLVs. Operations
       will be carried out sequentially starting from operation 0 towards
       ops_count-1 */
    pll_update_partials(treeinfo->partitions[p],
                        treeinfo->operations,
                        ops_count);

    pllmod_treeinfo_validate_clvs(treeinfo,
                                  treeinfo->travbuffer,
                                  traversal_size);

    /* compute the likelihood on an edge of the unrooted tree by specifying
       the CLV indices at the two end-point of the branch, the probability
       matrix index for the concrete branch length, and the index of the model
       of whose frequency vector is to be used */
    treeinfo->partition_loglh[p] = pll_compute_edge_loglikelihood(
                                            treeinfo->partitions[p],
                                            treeinfo->root->clv_index,
                                            treeinfo->root->scaler_index,
                                            treeinfo->root->back->clv_index,
                                            treeinfo->root->back->scaler_index,
                                            treeinfo->root->pmatrix_index,
                                            treeinfo->param_indices[p],
                                            persite_lnl ? persite_lnl[p] : NULL);
    
    // std::cout << "treeinfo->partition_loglh[p]: " << treeinfo->partition_loglh[p]  << std::endl;
  }

  /* sum up likelihood from all threads */
  if (treeinfo->parallel_reduce_cb)
  {
    treeinfo->parallel_reduce_cb(treeinfo->parallel_context,
                                 treeinfo->partition_loglh,
                                 p,
                                 PLLMOD_COMMON_REDUCE_SUM);
  }

  /* accumulate loglh by summing up over all the partitions */
  for (p = 0; p < treeinfo->partition_count; ++p)
    total_loglh += treeinfo->partition_loglh[p];

  /* restore original active partition */
  pllmod_treeinfo_set_active_partition(treeinfo, old_active_partition);

  assert(total_loglh < 0.);

  return total_loglh;
}


void check_tree(const PartitionedMSA& msa, const Tree& tree, bool require_binary)
{
  auto missing_taxa = 0;
  auto duplicate_taxa = 0;

  if (require_binary && !tree.binary())
  {
    throw runtime_error("Binary tree expected, but a tree with multifurcations provided!");
  }

  if (msa.taxon_count() > tree.num_tips())
    throw runtime_error("Alignment file contains more sequences than expected");
  else if (msa.taxon_count() != tree.num_tips())
    throw runtime_error("Some taxa are missing from the alignment file");

  unordered_set<string> tree_labels;
  unordered_set<string> msa_labels(msa.taxon_names().cbegin(), msa.taxon_names().cend());

  for (const auto& tip: tree.tip_labels())
  {
    if (!tree_labels.insert(tip.second).second)
    {
      LOG_ERROR << "ERROR: Taxon name appears more than once in the tree: " << tip.second << endl;
      duplicate_taxa++;
    }

    if (msa_labels.count(tip.second) == 0)
    {
      LOG_ERROR << "ERROR: Taxon name not found in the alignment: " << tip.second << endl;
      missing_taxa++;
    }
  }

  if (duplicate_taxa > 0)
    throw runtime_error("Tree contains duplicate taxon names (see above)!");

  if (missing_taxa > 0)
    throw runtime_error("Please check that sequence labels in the alignment and in the tree file are identical!");

  /* check for negative branch length */
  for (const auto& branch: tree.topology())
  {
    if (branch.length < 0.)
      throw runtime_error("Tree file contains negative branch lengths!");
  }
}


void prepare_tree(const RaxmlInstance& instance, Tree& tree)
{
  /* fix missing & outbound branch lengths */
  tree.fix_missing_brlens();
  tree.fix_outbound_brlens(instance.opts.brlen_min, instance.opts.brlen_max);

  /* make sure tip indices are consistent between MSA and pll_tree */
  assert(!instance.parted_msa->taxon_id_map().empty());
  tree.reset_tip_ids(instance.tip_id_map);
}


Tree generate_tree(const RaxmlInstance& instance, StartingTree type, int random_seed)
{
  Tree tree;

  const auto& opts = instance.opts;
  const auto& parted_msa = *instance.parted_msa;

  switch (type)
  {
    case StartingTree::user:
    {
      assert(instance.start_tree_stream);

      /* parse the unrooted binary tree in newick format, and store the number
         of tip nodes in tip_nodes_count */
      *instance.start_tree_stream >> tree;

      LOG_DEBUG << "Loaded user starting tree with " << tree.num_tips() << " taxa from: "
                           << opts.tree_file << endl;

      check_tree(parted_msa, tree, true);

      if (!instance.constraint_tree.empty())
      {
        tree.reset_tip_ids(instance.tip_id_map);
        if (!instance.constraint_tree.compatible(tree))
        {
          throw runtime_error("User starting tree incompatible with the specified topological constraint!");
        }
      }

      break;
    }
    case StartingTree::random:
      /* no starting tree provided, generate a random one */

      if (instance.constraint_tree.empty())
        tree = Tree::buildRandom(parted_msa.taxon_names(), random_seed);
      else
        tree = Tree::buildRandomConstrained(parted_msa.taxon_names(), random_seed,
                                            instance.constraint_tree);

      LOG_VERB_TS << "Generating a RANDOM starting tree, seed: " << random_seed << endl;

      break;
    case StartingTree::parsimony:
    {
      unsigned int score;

      const ParsimonyMSA& pars_msa = *instance.parted_msa_parsimony.get();
      tree = Tree::buildParsimonyConstrained(pars_msa, random_seed, &score,
                                             instance.constraint_tree, instance.tip_msa_idmap);

      LOG_WORKER_TS(LogLevel::verbose) << "Generated a PARSIMONY starting tree, seed: " << random_seed <<
          ", score: " << score << endl;

      break;
    }
    default:
      sysutil_fatal("Unknown starting tree type: %d\n", type);
  }

  assert(!tree.empty());

  prepare_tree(instance, tree);

  return tree;
}


size_t total_free_params(const RaxmlInstance& instance)
{
  const auto& parted_msa = *instance.parted_msa;
  size_t free_params = parted_msa.total_free_model_params();
  size_t num_parts = parted_msa.part_count();
  auto tree = BasicTree(parted_msa.taxon_count());
  auto num_branches = tree.num_branches();
  auto brlen_linkage = instance.opts.brlen_linkage;

  if (brlen_linkage == PLLMOD_COMMON_BRLEN_LINKED)
    free_params += num_branches;
  else if (brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED)
    free_params += num_branches + num_parts - 1;
  else if (brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED)
    free_params += num_branches * num_parts;

  return free_params;
}


void check_models(const RaxmlInstance& instance)
{
  const auto& opts = instance.opts;
  bool zero_freqs = false;

  for (const auto& pinfo: instance.parted_msa->part_list())
  {
    auto stats = pinfo.stats();
    auto model = pinfo.model();
    auto freq_mode = model.param_mode(PLLMOD_OPT_PARAM_FREQUENCIES);
    const auto& freqs = freq_mode == ParamValue::empirical ? stats.emp_base_freqs : model.base_freqs(0);

    // check for non-recommended model combinations
    if (opts.safety_checks.isset(SafetyCheck::model_lg4_freqs))
    {
      if ((model.name() == "LG4X" || model.name() == "LG4M") &&
          model.param_mode(PLLMOD_OPT_PARAM_FREQUENCIES) != ParamValue::model)
      {
        throw runtime_error("Partition \"" + pinfo.name() +
                            "\": You specified LG4M or LG4X model with shared stationary based frequencies (" +
                            model.to_string(false) + ").\n"
                            "Please be warned, that this is against the idea of LG4 models and hence it's not recommended!" + "\n"
                            "If you know what you're doing, you can add --force command line switch to disable this safety check.");
      }
    }

    // check for zero state frequencies
    if (opts.safety_checks.isset(SafetyCheck::model_zero_freqs))
    {
      if (freq_mode == ParamValue::empirical || freq_mode == ParamValue::user)
      {
        for (unsigned int i = 0; i < freqs.size(); ++i)
        {
          if (freqs[i] < PLL_EIGEN_MINFREQ)
          {
            if (!zero_freqs)
            {
              LOG_WARN << endl;
              zero_freqs = true;
            }

            LOG_WARN << "WARNING: State " << to_string(i) <<
                (instance.parted_msa->part_count() > 1 ? " in partition " + pinfo.name() : "") <<
                " has very low frequency (" << FMT_PREC9(freqs[i]) << ")!" << endl;

            LOG_VERB << "Base frequencies: ";
            for (unsigned int j = 0; j < freqs.size(); ++j)
              LOG_VERB << FMT_PREC9(freqs[j]) <<  " ";
            LOG_VERB << endl;
          }
        }
      }
    }

    // check for user-defined state frequencies which do not sum up to one
    if (opts.safety_checks.isset(SafetyCheck::model_invalid_freqs))
    {
      if (freq_mode == ParamValue::user)
      {
        double sum = 0.;
        for (unsigned int i = 0; i < freqs.size(); ++i)
          sum += freqs[i];

        if (fabs(sum - 1.0) > 0.01)
        {
          LOG_ERROR << "\nBase frequencies: ";
          for (unsigned int j = 0; j < freqs.size(); ++j)
            LOG_ERROR << FMT_PREC9(freqs[j]) <<  " ";
          LOG_ERROR << endl;

          throw runtime_error("User-specified stationary base frequencies"
                              " in partition " + pinfo.name() + " do not sum up to 1.0!\n"
                              "Please provide normalized frequencies.");
        }
      }
    }

    if (model.num_submodels() > 1 &&
        (model.param_mode(PLLMOD_OPT_PARAM_FREQUENCIES) == ParamValue::ML ||
         model.param_mode(PLLMOD_OPT_PARAM_SUBST_RATES) == ParamValue::ML))
    {
      throw runtime_error("Invalid model " + model.to_string(false) + " in partition " + pinfo.name() + ":\n"
                          "Mixture models with ML estimates of rates/frequencies are not supported yet!");
    }

    // check partitions which contain invariant sites and have ascertainment bias enabled
    if (opts.safety_checks.isset(SafetyCheck::model_asc_bias))
    {
      if (model.ascbias_type() != AscBiasCorrection::none && stats.inv_count() > 0)
      {
        throw runtime_error("You enabled ascertainment bias correction for partition " +
                             pinfo.name() + ", but it contains " +
                             to_string(stats.inv_count()) + " invariant sites.\n"
                            "This is not allowed! Please either remove invariant sites from MSA "
                            "or disable ascertainment bias correction.");
      }
    }
  }

  if (zero_freqs)
  {
    LOG_WARN << endl << "WARNING: Some states have very low frequencies, "
        "which might lead to numerical issues!" << endl;
  }

  /* Check for extreme cases of overfitting (K >= n) */
  if (opts.safety_checks.isset(SafetyCheck::model_overfit))
  {
    if (instance.parted_msa->part_count() > 1)
    {
      size_t model_free_params = instance.parted_msa->total_free_model_params();
      size_t free_params = total_free_params(instance);
      size_t sample_size = instance.parted_msa->total_sites();
      string errmsg = "Number of free parameters (K=" + to_string(free_params) +
          ") is larger than alignment size (n=" + to_string(sample_size) + ").\n" +
          "       This might lead to overfitting and compromise tree inference results!\n" +
          "       Please consider revising your partitioning scheme, conducting formal model selection\n" +
          "       and/or using linked/scaled branch lengths across partitions.\n" +
          "NOTE:  You can disable this check by adding the --force option.\n";

      if (free_params >= sample_size)
      {
        if (model_free_params >= sample_size ||
            instance.opts.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED)
        {
          throw runtime_error(errmsg);
        }
        else
          LOG_WARN << endl << "WARNING: " << errmsg << endl;
      }
    }
  }
}


void check_options_early(Options& opts)
{
  auto num_procs = opts.num_ranks * (opts.num_threads > 0 ? opts.num_threads : opts.num_threads_max);

  if (!opts.weights_file.empty() && !sysutil_file_exists(opts.weights_file))
    throw runtime_error("Site weights file not found: " + opts.weights_file);

  if (!opts.constraint_tree_file.empty() &&
      ((opts.start_trees.count(StartingTree::parsimony) > 0 ||
       opts.start_trees.count(StartingTree::user) > 0) && opts.use_old_constraint))
  {
    throw runtime_error(" User and parsimony starting trees are not supported in combination with "
                        "constrained tree inference.\n"
                        "       Please use random starting trees instead.");
  }

  if (opts.num_workers > num_procs)
  {
    throw OptionException("The specified number of parallel tree searches (" +
                          to_string(opts.num_workers) +
                          ") is higher than the number of available threads (" +
                          to_string(num_procs) + ")");
  }

  /* check for unsupported coarse-grained topology */
  if (opts.coarse() && opts.num_ranks > opts.num_workers)
  {
    throw runtime_error("Unsupported parallelization topology!\n"
                        "NOTE:  Multiple MPI ranks per worker are not allowed in coarse-grained mode.\n");
  }

  if (opts.coarse() && (opts.num_ranks * opts.num_threads % opts.num_workers != 0))
  {
    throw OptionException("The specified number of threads (" +
                          to_string(opts.num_ranks * opts.num_threads) +
                          ") is not a multiple of the number of parallel tree searches (" +
                          to_string(opts.num_workers) + ")\n" +
                          "HINT:  Consider using --workers auto{" + to_string(opts.num_workers) + "}");
  }

  /* writing interim result files is not supported in coarse+MPI mode -> too much hassle */
  if (opts.coarse() && opts.num_ranks > 1)
    opts.write_interim_results = false;

  if (opts.command == Command::ancestral)
  {
    if (opts.use_pattern_compression)
      throw runtime_error("Pattern compression is not supported in ancestral state reconstruction mode!");
    if (opts.use_repeats)
      throw runtime_error("Site repeats are not supported in ancestral state reconstruction mode!");
    if (opts.use_rate_scalers)
      throw runtime_error("Per-rate scalers are not supported in ancestral state reconstruction mode!");
    if (opts.num_ranks > 1)
      throw runtime_error("MPI parallelization is not supported in ancestral state reconstruction mode!");
  }

  if (opts.command == Command::sitelh)
  {
    if (opts.num_ranks > 1)
      throw runtime_error("MPI parallelization is not supported in per-site likelihood computation mode!");
    if (!opts.weights_file.empty())
      throw runtime_error("Custom site weights are not supported in per-site likelihood computation mode!");
  }

  /* autodetect if we can use partial RBA loading */
  opts.use_rba_partload &= (opts.num_ranks > 1 && !opts.coarse());                // only useful for fine-grain MPI runs
  opts.use_rba_partload &= (!opts.start_trees.count(StartingTree::parsimony));    // does not work with parsimony
  opts.use_rba_partload &= (opts.command == Command::search ||                    // currently doesn't work with bootstrap
                            opts.command == Command::evaluate ||
                            opts.command == Command::ancestral);

  LOG_DEBUG << "RBA partial loading: " << (opts.use_rba_partload ? "ON" : "OFF") << endl;
}


void check_options(RaxmlInstance& instance)
{
  const auto& opts = instance.opts;

  /* check that all outgroup taxa are present in the alignment */
  if (!opts.outgroup_taxa.empty())
  {
    NameList missing_taxa;
    for (const auto& ot: opts.outgroup_taxa)
    {
      if (!instance.parted_msa->taxon_id_map().count(ot))
        missing_taxa.push_back(ot);
    }

    if (!missing_taxa.empty())
    {
      LOG_ERROR << "ERROR: Following taxa were specified as an outgroup "
                                                     "but are missing from the alignment:" << endl;
      for (const auto& mt: missing_taxa)
        LOG_ERROR << mt << endl;
      LOG_ERROR << endl;
      throw runtime_error("Outgroup taxon not found.");
    }
  }

  /* check that we have enough patterns per thread */
  if (opts.safety_checks.isset(SafetyCheck::perf_threads))
  {
    if (ParallelContext::master_rank() && ParallelContext::num_procs() > 1)
    {
      StaticResourceEstimator resEstimator(*instance.parted_msa, instance.opts);
      auto res = resEstimator.estimate();
      if (ParallelContext::threads_per_group() > res.num_threads_response)
      {
        LOG_WARN << endl;
        LOG_WARN << "WARNING: You might be using too many threads (" << ParallelContext::num_procs()
                 <<  ") for your alignment with "
                 << (opts.use_pattern_compression ?
                        to_string(instance.parted_msa->total_patterns()) + " unique patterns." :
                        to_string(instance.parted_msa->total_sites()) + " alignment sites.")
                 << endl;
        LOG_WARN << "NOTE:    For the optimal throughput, please consider using fewer threads " << endl;
        LOG_WARN << "NOTE:    and parallelize across starting trees/bootstrap replicates." << endl;
        LOG_WARN << "NOTE:    As a general rule-of-thumb, please assign at least 200-1000 "
            "alignment patterns per thread." << endl << endl;

        if (ParallelContext::threads_per_group() > 2 * res.num_threads_response)
        {
          throw runtime_error("Too few patterns per thread! "
                              "RAxML-NG will terminate now to avoid wasting resources.\n"
                              "NOTE:  Please reduce the number of threads (see guidelines above).\n"
                              "NOTE:  This check can be disabled with the '--force perf_threads' option.");
        }
      }
    }
  }

  /* auto-enable rate scalers for >2000 taxa */
  if (opts.safety_checks.isset(SafetyCheck::model_rate_scalers))
  {
    if (instance.parted_msa->taxon_count() > RAXML_RATESCALERS_TAXA &&
        !instance.opts.use_rate_scalers && opts.command != Command::ancestral)
    {
      LOG_INFO << "\nNOTE: Per-rate scalers were automatically enabled to prevent numerical issues "
          "on taxa-rich alignments." << endl;
      LOG_INFO << "NOTE: You can use --force switch to skip this check "
          "and fall back to per-site scalers." << endl << endl;
      instance.opts.use_rate_scalers = true;
    }
  }

  /* make sure we do not check for convergence too often in coarse-grained parallelization mode */
  instance.opts.bootstop_interval = std::max(opts.bootstop_interval, opts.num_workers*2);
}


bool check_msa_global(const MSA& msa)
{
  bool msa_valid = true;

  /* check taxa count */
  if (msa.size() < 4)
  {
    LOG_ERROR << "\nERROR: Your alignment contains less than 4 sequences! " << endl;
    msa_valid = false;
  }

  /* check for duplicate taxon names */
  unsigned long stats_mask = PLLMOD_MSA_STATS_DUP_TAXA;

  pllmod_msa_stats_t * stats = pllmod_msa_compute_stats(msa.pll_msa(),
                                                        4,
                                                        pll_map_nt, // map is not used here
                                                        NULL,
                                                        stats_mask);

  libpll_check_error("ERROR computing MSA stats");
  assert(stats);

  if (stats->dup_taxa_pairs_count > 0)
  {
    LOG_ERROR << endl;
    for (unsigned long c = 0; c < stats->dup_taxa_pairs_count; ++c)
    {
      auto id1 = stats->dup_taxa_pairs[c*2];
      auto id2 = stats->dup_taxa_pairs[c*2+1];
      LOG_ERROR << "ERROR: Sequences " << id1+1 << " and "
                << id2+1 << " have identical name: "
                << msa.label(id1) << endl;
    }
    LOG_ERROR << "\nERROR: Duplicate sequence names found: "
              << stats->dup_taxa_pairs_count << endl;

    msa_valid = false;
  }

  pllmod_msa_destroy_stats(stats);

  return msa_valid;
}


bool check_msa(RaxmlInstance& instance)
{
  LOG_VERB_TS << "Checking the alignment...\n";

  const auto& opts = instance.opts;
  auto& parted_msa = *instance.parted_msa;
  const auto& full_msa = parted_msa.full_msa();
  const auto pll_msa = full_msa.pll_msa();

  bool msa_valid = true;
  bool msa_corrected = false;
  PartitionedMSAView parted_msa_view(instance.parted_msa);

  vector<pair<size_t,size_t> > dup_taxa;
  vector<pair<size_t,size_t> > dup_seqs;
  std::set<size_t> gap_seqs;

  /* check taxon names for invalid characters */
  if (opts.safety_checks.isset(SafetyCheck::msa_names))
  {
    const string invalid_chars = "[](),;:' \t\n";
    for (const auto& taxon: parted_msa.taxon_names())
    {
      if (taxon.find_first_of(invalid_chars) != std::string::npos)
      {
        size_t i = 0;
        auto fixed_name = taxon;
        while ((i = fixed_name.find_first_of(invalid_chars, i)) != std::string::npos)
          fixed_name[i++] = '_';
        parted_msa_view.map_taxon_name(taxon, fixed_name);
      }
    }

    msa_valid &= parted_msa_view.taxon_name_map().empty();
  }

  /* check for duplicate sequences */
  if (opts.safety_checks.isset(SafetyCheck::msa_dups))
  {
    unsigned long stats_mask = PLLMOD_MSA_STATS_DUP_SEQS;

    pllmod_msa_stats_t * stats = pllmod_msa_compute_stats(pll_msa,
                                                          4,
                                                          pll_map_nt, // map is not used here
                                                          NULL,
                                                          stats_mask);

    libpll_check_error("ERROR computing MSA stats");
    assert(stats);

    for (unsigned long c = 0; c < stats->dup_seqs_pairs_count; ++c)
    {
      dup_seqs.emplace_back(stats->dup_seqs_pairs[c*2],
                            stats->dup_seqs_pairs[c*2+1]);
    }

    pllmod_msa_destroy_stats(stats);
  }

  size_t total_gap_cols = 0;
  size_t part_num = 0;
  for (auto& pinfo: parted_msa.part_list())
  {
    /* check for invalid MSA characters */
    pllmod_msa_errors_t * errs = pllmod_msa_check(pinfo.msa().pll_msa(),
                                                  pinfo.model().charmap());

    if (errs)
    {
      if (errs->invalid_char_count > 0)
      {
        msa_valid = false;
        LOG_ERROR << endl;
        for (unsigned long c = 0; c < errs->invalid_char_count; ++c)
        {
          auto global_pos = parted_msa.full_msa_site(part_num, errs->invalid_char_pos[c]);
          LOG_ERROR << "ERROR: Invalid character in sequence " <<  errs->invalid_char_seq[c]+1
                    << " at position " <<  global_pos+1  << ": " << errs->invalid_chars[c] << endl;
        }
        part_num++;
        continue;
      }
      pllmod_msa_destroy_errors(errs);
    }
    else
      libpll_check_error("MSA check failed");


    /* Check for all-gap columns and sequences */
    if (opts.safety_checks.isset(SafetyCheck::msa_allgaps))
    {
      unsigned long stats_mask = PLLMOD_MSA_STATS_GAP_SEQS | PLLMOD_MSA_STATS_GAP_COLS;

      pllmod_msa_stats_t * stats = pinfo.compute_stats(stats_mask);

      if (stats->gap_cols_count > 0)
      {
        total_gap_cols += stats->gap_cols_count;
        std::vector<size_t> gap_cols(stats->gap_cols, stats->gap_cols + stats->gap_cols_count);
        pinfo.msa().remove_sites(gap_cols);
  //      parted_msa_view.exclude_sites(part_num, gap_cols);
      }

      std::set<size_t> cur_gap_seq(stats->gap_seqs, stats->gap_seqs + stats->gap_seqs_count);

      if (!part_num)
      {
        gap_seqs = cur_gap_seq;
      }
      else
      {
        for(auto it = gap_seqs.begin(); it != gap_seqs.end();)
        {
          if(cur_gap_seq.find(*it) == cur_gap_seq.end())
            it = gap_seqs.erase(it);
          else
            ++it;
        }
      }

      pllmod_msa_destroy_stats(stats);
    }

    part_num++;
  }

  if (total_gap_cols > 0)
  {
    LOG_WARN << "\nWARNING: Fully undetermined columns found: " << total_gap_cols << endl;
    msa_corrected = true;
  }

  if (!gap_seqs.empty())
  {
   LOG_WARN << endl;
   for (auto c : gap_seqs)
   {
     parted_msa_view.exclude_taxon(c);
     LOG_VERB << "WARNING: Sequence #" << c+1 << " (" << parted_msa.taxon_names().at(c)
              << ") contains only gaps!" << endl;
   }
   LOG_WARN << "WARNING: Fully undetermined sequences found: " << gap_seqs.size() << endl;
  }

  if (!dup_seqs.empty())
  {
    size_t dup_count = 0;
    LOG_WARN << endl;
    for (const auto& p: dup_seqs)
    {
      /* ignore gap-only sequences */
      if (gap_seqs.count(p.first) || gap_seqs.count(p.second))
        continue;

      ++dup_count;
      parted_msa_view.exclude_taxon(p.second);
      LOG_WARN << "WARNING: Sequences " << parted_msa.taxon_names().at(p.first) << " and " <<
          parted_msa.taxon_names().at(p.second) << " are exactly identical!" << endl;
    }
    if (dup_count > 0)
      LOG_WARN << "WARNING: Duplicate sequences found: " << dup_count << endl;
  }

  if (!instance.opts.nofiles_mode && (msa_corrected || !parted_msa_view.identity()))
  {
    // print_reduced_msa(instance, parted_msa_view);
  }

  if (!parted_msa_view.taxon_name_map().empty())
  {
    LOG_ERROR << endl;
    for (auto& it: parted_msa_view.taxon_name_map())
      LOG_ERROR << "ERROR: Following taxon name contains invalid characters: " << it.first << endl;

    LOG_ERROR << endl;
    LOG_INFO << "NOTE: Following symbols are not allowed in taxa names to ensure Newick compatibility:\n"
                "NOTE: \" \" (space), \";\" (semicolon), \":\" (colon), \",\" (comma), "
                       "\"()\" (parentheses), \"'\" (quote). " << endl;
    LOG_INFO << "NOTE: Please either correct the names manually, or use the reduced alignment file\n"
                "NOTE: generated by RAxML-NG (see above).";
    LOG_INFO << endl;
  }

  return msa_valid;
}


void build_parsimony_msa(RaxmlInstance& instance)
{
  unsigned int attrs = instance.opts.simd_arch;

  // TODO: check if there is any reason not to use tip-inner
  attrs |= PLL_ATTRIB_PATTERN_TIP;

  instance.parted_msa_parsimony.reset(new ParsimonyMSA(instance.parted_msa, attrs));
}


void check_oversubscribe(RaxmlInstance& instance)
{
  const auto& opts = instance.opts;
  if (opts.safety_checks.isset(SafetyCheck::perf_threads))
  {
    size_t iters = 100;
    auto start = global_timer().elapsed_seconds();
    for (size_t i = 0; i < iters; ++i)
       ParallelContext::global_barrier();

    if (ParallelContext::master())
    {
      double sync_time = 1000. * (global_timer().elapsed_seconds() - start) / iters;

      LOG_DEBUG << endl << "BARRIER time: " << FMT_PREC6(sync_time) << " ms" << endl;

      /* empirical threshold: >5ms per barrier looks suspicious */
      if (sync_time > 5. + 0.1 * log2(ParallelContext::num_nodes()))
      {
          throw runtime_error("CPU core oversubscription detected! "
                              "RAxML-NG will terminate now to avoid wasting resources.\n"
                              "NOTE:  Details: https://github.com/amkozlov/raxml-ng/wiki/Parallelization#core-oversubscription\n"
                              "NOTE:  You can use '--force perf_threads' to disable this check, "
                              "but ONLY if you are 200% sure this is a false alarm!");
      }
    }
  }
}


void build_start_trees(RaxmlInstance& instance, unsigned int num_threads = 1)
{
  auto& opts = instance.opts;
  const auto& parted_msa = *instance.parted_msa;

  /* all start trees were already generated/loaded -> return */
  if (instance.start_trees.size() >= instance.opts.num_searches)
    return;

  for (auto& st_tree: opts.start_trees)
  {
    auto st_tree_type = st_tree.first;
    auto& st_tree_count = st_tree.second;

    // init seeds
    intVector seeds(st_tree_count);
    for (size_t i = 0; i < st_tree_count; ++i)
      seeds[i] = rand();

    switch (st_tree_type)
    {
      case StartingTree::user:
        LOG_INFO_TS << "Loading user starting tree(s) from: " << opts.tree_file << endl;
        if (!sysutil_file_exists(opts.tree_file))
          throw runtime_error("File not found: " + opts.tree_file);
        instance.start_tree_stream.reset(new NewickStream(opts.tree_file, std::ios::in));
        break;
      case StartingTree::random:
        LOG_INFO_TS << "Generating " << st_tree_count << " random starting tree(s) with "
                    << parted_msa.taxon_count() << " taxa" << endl;
        break;
      case StartingTree::parsimony:
        build_parsimony_msa(instance);
        LOG_INFO_TS << "Generating " << st_tree_count << " parsimony starting tree(s) with "
                    << parted_msa.taxon_count() << " taxa" << endl;
        break;
      default:
        assert(0);
    }

    if (num_threads != 1 && st_tree_type == StartingTree::parsimony)
    {

    }
    else
    {
      for (size_t i = 0; i < st_tree_count; ++i)
      {
        auto tree = generate_tree(instance, st_tree_type, seeds[i]);

        // TODO use universal starting tree generator
        if (st_tree_type == StartingTree::user)
        {
          if (instance.start_tree_stream->peek() != EOF)
          {
            st_tree_count++;
            opts.num_searches++;
          }
        }

        instance.start_trees.emplace_back(tree);
      }
    }

  }

  // free memory used for parsimony MSA
  instance.parted_msa_parsimony.release();

  if (::ParallelContext::master_rank())
  {
    NewickStream nw_start(opts.start_tree_file());
    for (auto const& tree: instance.start_trees)
      nw_start << tree;
  }
}


void balance_load(RaxmlInstance& instance)
{
  PartitionAssignment part_sizes;

  /* init list of partition sizes */
  size_t i = 0;
  for (auto const& pinfo: instance.parted_msa->part_list())
  {
    part_sizes.assign_sites(i, 0, pinfo.length(), pinfo.model().clv_entry_size());
    ++i;
  }

  instance.proc_part_assign =
      instance.load_balancer->get_all_assignments(part_sizes, ParallelContext::threads_per_group());

  LOG_INFO_TS << "Data distribution: " << PartitionAssignmentStats(instance.proc_part_assign) << endl;
  LOG_VERB << endl << instance.proc_part_assign;
}


void init_part_info(RaxmlInstance& instance)
{
  auto& opts = instance.opts;

  instance.parted_msa = std::make_shared<PartitionedMSA>();
  auto& parted_msa = *instance.parted_msa;

  if (!sysutil_file_exists(opts.msa_file))
  {
    throw runtime_error("Alignment file not found: " + opts.msa_file);
  }

  /* check if we have a binary input file */
  if (opts.msa_format == FileFormat::binary ||
      (opts.msa_format == FileFormat::autodetect && RBAStream::rba_file(opts.msa_file)))
  {
    opts.msa_format = FileFormat::binary;

    if (opts.command == Command::sitelh)
    {
      throw runtime_error("Alignments in RBA format are not supported in "
          "per-site likelihood mode, sorry!\n       Please use PHYLIP/FASTA instead.");
    }

    if (!opts.model_file.empty())
    {
      LOG_WARN <<
          "WARNING: The model you specified on the command line (" << opts.model_file <<
                    ") will be ignored " << endl <<
          "         since the binary MSA file already contains a model definition." << endl <<
          "         If you want to change the model, please re-run RAxML-NG "  << endl <<
          "         with the original PHYLIP/FASTA alignment and --redo option."
          << endl << endl;
    }

    if (!opts.weights_file.empty())
    {
      LOG_WARN <<
          "WARNING: Alignment site weights file (" << opts.weights_file <<
                    ") will be ignored!" << endl <<
          "NOTE:    Custom site weights are not allowed in combination with RBA input."
          << endl << endl;
    }

    LOG_INFO_TS << "Loading binary alignment from file: " << opts.msa_file << endl;

    auto rba_elem = opts.use_rba_partload ? RBAStream::RBAElement::metadata : RBAStream::RBAElement::all;
    RBAStream bs(opts.msa_file);
    bs >> RBAStream::RBAOutput(parted_msa, rba_elem, nullptr);

    // binary probMSAs are not supported yet
    instance.opts.use_prob_msa = false;

    LOG_INFO_TS << "Alignment comprises " << parted_msa.taxon_count() << " taxa, " <<
        parted_msa.part_count() << " partitions and " <<
        parted_msa.total_length() << " patterns\n" << endl;

    LOG_INFO << parted_msa;

    LOG_INFO << endl;
  }
  /* check if model is a file */
  else if (sysutil_file_exists(opts.model_file))
  {
    // read partition definitions from file
    try
    {
      RaxmlPartitionStream partfile(opts.model_file, ios::in);
      partfile >> parted_msa;
    }
    catch(exception& e)
    {
      throw runtime_error("Failed to read partition file:\n" + string(e.what()));
    }
  }
  else if (!opts.model_file.empty())
  {
    // create and init single pseudo-partition
    parted_msa.emplace_part_info("noname", opts.data_type, opts.model_file);
  }
  else
    throw runtime_error("Please specify an evolutionary model with --model switch");

  assert(parted_msa.part_count() > 0);

  /* make sure that linked branch length mode is set for unpartitioned alignments */
  if (parted_msa.part_count() == 1)
  {
    opts.brlen_linkage = PLLMOD_COMMON_BRLEN_LINKED;
    if (opts.safety_checks.isset(SafetyCheck::model) &&
        parted_msa.model(0).param_mode(PLLMOD_OPT_PARAM_BRANCH_LEN_SCALER) != ParamValue::undefined)
      throw runtime_error("Branch length scalers (+B) are not supported for non-partitioned models!");
  }

  /* in the scaled brlen mode, use ML optimization of brlen scalers by default */
  if (opts.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED)
  {
    for (auto& pinfo: parted_msa.part_list())
      pinfo.model().set_param_mode_default(PLLMOD_OPT_PARAM_BRANCH_LEN_SCALER, ParamValue::ML);
  }

  int freerate_count = 0;

  for (const auto& pinfo: parted_msa.part_list())
  {
    LOG_DEBUG << "|" << pinfo.name() << "|   |" << pinfo.model().to_string() << "|   |" <<
        pinfo.range_string() << "|" << endl;

    if (pinfo.model().ratehet_mode() == PLLMOD_UTIL_MIXTYPE_FREE)
      freerate_count++;
  }

  if (parted_msa.part_count() > 1 && freerate_count > 0 &&
      opts.brlen_linkage == PLLMOD_COMMON_BRLEN_LINKED)
  {
    throw runtime_error("LG4X and FreeRate models are not supported in linked branch length mode.\n"
        "Please use the '--brlen scaled' option to switch into proportional branch length mode.");
  }
}


void load_msa_weights(MSA& msa, const Options& opts)
{
  /* load site weights from file */
  if (!opts.weights_file.empty())
  {
    LOG_VERB_TS << "Loading site weights... " << endl;

    /* RBA file contains collapsed site weights -> not compatible with weights file! */
    assert(opts.msa_format != FileFormat::binary);

    FILE* f = fopen(opts.weights_file.c_str(), "r");
    if (!f)
      throw runtime_error("Unable to open site weights file: " + opts.weights_file);

    WeightVector w;
    w.reserve(msa.length());
    const auto maxw = std::numeric_limits<WeightVector::value_type>::max();
    int fres;
    intmax_t x;
    while ((fres = fscanf(f,"%jd", &x)) != EOF)
    {
      if (!fres)
      {
        char buf[101];
        size_t c = fread(buf, 1, 100, f);
        buf[c] = 0;
        fclose(f);
        throw runtime_error("Invalid site weight entry found near: " + string(buf));
      }
      else if (x <= 0)
      {
        fclose(f);
        throw runtime_error("Non-positive site weight found: " + to_string(x) +
                            " (at position " + to_string(w.size()+1) + ")");
      }
      else if (x > maxw)
      {
        fclose(f);
        throw runtime_error("Site weight too large: " + to_string(x) +
                            " (max: " + to_string(maxw) + ")");
      }
      else
        w.push_back((WeightType) x);
    }
    fclose(f);

    if (w.size() != msa.length())
    {
      throw runtime_error("Site weights file contains the wrong number of entries: " +
                          to_string(w.size()) + " (expected: " + to_string(msa.length()) + ")" +
                          "\nPlease check that this file contains one positive integer per site: "
                          + opts.weights_file);
    }
    msa.weights(w);
  }
}


void load_msa(RaxmlInstance& instance)
{
  const auto& opts = instance.opts;
  auto& parted_msa = *instance.parted_msa;

  LOG_INFO_TS << "Reading alignment from file: " << opts.msa_file << endl;

  /* load MSA */
  auto msa = msa_load_from_file(opts.msa_file, opts.msa_format);

  if (!msa.size())
    throw runtime_error("Alignment file is empty!");

  LOG_INFO_TS << "Loaded alignment with " << msa.size() << " taxa and " <<
      msa.num_sites() << " sites" << endl;

  if (msa.probabilistic() && opts.use_prob_msa)
  {
    instance.opts.use_pattern_compression = false;
    instance.opts.use_tip_inner = false;
    instance.opts.use_repeats = false;

    if (parted_msa.part_count() > 1)
      throw runtime_error("Partitioned probabilistic alignments are not supported yet, sorry...");
  }
  else
    instance.opts.use_prob_msa = false;

  if (!check_msa_global(msa))
    throw runtime_error("Alignment check failed (see details above)!");

  load_msa_weights(msa, opts);

  parted_msa.full_msa(std::move(msa));

  LOG_VERB_TS << "Extracting partitions... " << endl;

  parted_msa.split_msa();

  /* check alignment */
  if (!check_msa(instance))
    throw runtime_error("Alignment check failed (see details above)!");

  if (opts.use_pattern_compression)
  {
    LOG_VERB_TS << "Compressing alignment patterns... " << endl;
    bool store_backmap = opts.command == Command::sitelh;
    parted_msa.compress_patterns(store_backmap);

    // temp workaround: since MSA pattern compression calls rand(), it will change all random
    // numbers generated afterwards. so just reset seed to the initial value to ensure that
    // starting trees, BS replicates etc. are the same regardless whether pat.comp is ON or OFF
    srand(opts.random_seed);
  }

//  if (parted_msa.part_count() > 1)
//    instance.terrace_wrapper.reset(new TerraceWrapper(parted_msa));

  parted_msa.set_model_empirical_params();

  check_models(instance);

  LOG_INFO << endl;

  LOG_INFO << "Alignment comprises " << parted_msa.part_count() << " partitions and "
           << parted_msa.total_length() << (opts.use_pattern_compression ? " patterns" : " sites")
           << endl << endl;

  LOG_INFO << parted_msa;

  LOG_INFO << endl;

  if (ParallelContext::master_rank() &&
      !instance.opts.use_prob_msa && !instance.opts.binary_msa_file().empty())
  {
    auto binary_msa_fname = instance.opts.binary_msa_file();
    if (sysutil_file_exists(binary_msa_fname) && !opts.redo_mode &&
        opts.command != Command::parse)
    {
      LOG_INFO << "NOTE: Binary MSA file already exists: " << binary_msa_fname << endl << endl;
    }
    else if (opts.command != Command::check)
    {
      RBAStream bs(binary_msa_fname);
      bs << parted_msa;
      LOG_INFO << "NOTE: Binary MSA file created: " << binary_msa_fname << endl << endl;
    }
  }
}


void load_parted_msa(RaxmlInstance& instance)
{
  init_part_info(instance);

  assert(instance.parted_msa);

  if (instance.opts.msa_format != FileFormat::binary)
    load_msa(instance);

  // use MSA sequences IDs as "normalized" tip IDs in all trees
  instance.tip_id_map = instance.parted_msa->taxon_id_map();
}




void init_part_info_dsz(RaxmlInstance& instance)
{
  auto& opts = instance.opts;

  instance.parted_msa = std::make_shared<PartitionedMSA>();
  auto& parted_msa = *instance.parted_msa;

  // if (!sysutil_file_exists(opts.msa_file))
  // {
  //   throw runtime_error("Alignment file not found: " + opts.msa_file);
  // }

  /* check if we have a binary input file */
  if (opts.msa_format == FileFormat::binary ||
      (opts.msa_format == FileFormat::autodetect && RBAStream::rba_file(opts.msa_file)))
  {
    opts.msa_format = FileFormat::binary;

    if (opts.command == Command::sitelh)
    {
      throw runtime_error("Alignments in RBA format are not supported in "
          "per-site likelihood mode, sorry!\n       Please use PHYLIP/FASTA instead.");
    }

    if (!opts.model_file.empty())
    {
      LOG_WARN <<
          "WARNING: The model you specified on the command line (" << opts.model_file <<
                    ") will be ignored " << endl <<
          "         since the binary MSA file already contains a model definition." << endl <<
          "         If you want to change the model, please re-run RAxML-NG "  << endl <<
          "         with the original PHYLIP/FASTA alignment and --redo option."
          << endl << endl;
    }

    if (!opts.weights_file.empty())
    {
      LOG_WARN <<
          "WARNING: Alignment site weights file (" << opts.weights_file <<
                    ") will be ignored!" << endl <<
          "NOTE:    Custom site weights are not allowed in combination with RBA input."
          << endl << endl;
    }

    LOG_INFO_TS << "Loading binary alignment from file: " << opts.msa_file << endl;

    auto rba_elem = opts.use_rba_partload ? RBAStream::RBAElement::metadata : RBAStream::RBAElement::all;
    RBAStream bs(opts.msa_file);
    bs >> RBAStream::RBAOutput(parted_msa, rba_elem, nullptr);

    // binary probMSAs are not supported yet
    instance.opts.use_prob_msa = false;

    LOG_INFO_TS << "Alignment comprises " << parted_msa.taxon_count() << " taxa, " <<
        parted_msa.part_count() << " partitions and " <<
        parted_msa.total_length() << " patterns\n" << endl;

    LOG_INFO << parted_msa;

    LOG_INFO << endl;
  }
  /* check if model is a file */
  else if (sysutil_file_exists(opts.model_file))
  {
    // read partition definitions from file
    try
    {
      RaxmlPartitionStream partfile(opts.model_file, ios::in);
      partfile >> parted_msa;
    }
    catch(exception& e)
    {
      throw runtime_error("Failed to read partition file:\n" + string(e.what()));
    }
  }
  else if (!opts.model_file.empty())
  {
    // create and init single pseudo-partition
    parted_msa.emplace_part_info("noname", opts.data_type, opts.model_file);
  }
  else
    throw runtime_error("Please specify an evolutionary model with --model switch");

  assert(parted_msa.part_count() > 0);

  /* make sure that linked branch length mode is set for unpartitioned alignments */
  if (parted_msa.part_count() == 1)
  {
    opts.brlen_linkage = PLLMOD_COMMON_BRLEN_LINKED;
    if (opts.safety_checks.isset(SafetyCheck::model) &&
        parted_msa.model(0).param_mode(PLLMOD_OPT_PARAM_BRANCH_LEN_SCALER) != ParamValue::undefined)
      throw runtime_error("Branch length scalers (+B) are not supported for non-partitioned models!");
  }

  /* in the scaled brlen mode, use ML optimization of brlen scalers by default */
  if (opts.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED)
  {
    for (auto& pinfo: parted_msa.part_list())
      pinfo.model().set_param_mode_default(PLLMOD_OPT_PARAM_BRANCH_LEN_SCALER, ParamValue::ML);
  }

  int freerate_count = 0;

  for (const auto& pinfo: parted_msa.part_list())
  {
    LOG_DEBUG << "|" << pinfo.name() << "|   |" << pinfo.model().to_string() << "|   |" <<
        pinfo.range_string() << "|" << endl;

    if (pinfo.model().ratehet_mode() == PLLMOD_UTIL_MIXTYPE_FREE)
      freerate_count++;
  }

  if (parted_msa.part_count() > 1 && freerate_count > 0 &&
      opts.brlen_linkage == PLLMOD_COMMON_BRLEN_LINKED)
  {
    throw runtime_error("LG4X and FreeRate models are not supported in linked branch length mode.\n"
        "Please use the '--brlen scaled' option to switch into proportional branch length mode.");
  }
}




int argc = 15;
char *argv[15] = {
    "./raxml-ng",
    "--evaluate",
    "--msa",
    "xxxx.phy",
    "--threads",
    "1",
    "--model",
    "JC",
    "--tree",
    "xxxx.tre",
    "--prefix",
    "xxx",
    "--redo",
    "--opt-model",
    "off"
};



void build_raxml_instance(
    RaxmlInstance &instance,
    const std::string &tree_str,
    const std::vector<std::string> &labels,
    const std::vector<std::string> &sequences,
    bool tree_rooted
) {
    auto& opts = instance.opts;
    CommandLineParser cmdline;

    
    cmdline.parse_options(argc, ((char**) argv), opts);

    // check_options_early(opts);


    switch (opts.bootstop_criterion)
    {
      case BootstopCriterion::autoMRE:
        instance.bootstop_checker.reset(new BootstopCheckMRE(opts.num_bootstraps,
                                                             opts.bootstop_cutoff,
                                                             opts.bootstop_permutations));
        break;
      case BootstopCriterion::none:
        break;
      default:
        throw runtime_error("Only autoMRE bootstopping criterion is supported for now, sorry!");
    }


    // func master_main
    /* init load balancer */
    assert (opts.load_balance_method == LoadBalancing::benoit);
    switch(opts.load_balance_method)
    {
        case LoadBalancing::naive:
        instance.load_balancer.reset(new SimpleLoadBalancer());
        break;
        case LoadBalancing::kassian:
        instance.load_balancer.reset(new KassianLoadBalancer());
        break;
        case LoadBalancing::benoit:
        instance.load_balancer.reset(new BenoitLoadBalancer());
        break;
        default:
        assert(0);
    }

    // use naive coarse-grained load balancer for now
    instance.coarse_load_balancer.reset(new SimpleCoarseLoadBalancer());

    // load_parted_msa(instance);
    init_part_info_dsz(instance);
    assert(instance.parted_msa);



#if 0
    load_msa(instance);
    instance.tip_id_map = instance.parted_msa->taxon_id_map();

#else
    // load_msa(instance);
    // auto msa = msa_load_from_file(opts.msa_file, opts.msa_format);
    assert (opts.msa_format == FileFormat::autodetect);

    pll_msa_t * pll_msa = (pll_msa_t *) malloc(sizeof(pll_msa_t));
    pll_msa->count = labels.size();
    pll_msa->length = sequences[0].size();

    pll_msa->label = (char **) malloc(pll_msa->count * sizeof(char *));
    pll_msa->sequence = (char **) malloc(pll_msa->count * sizeof(char *));

    for (int i = 0; i < labels.size(); ++i) {
        pll_msa->label[i] = strdup(labels[i].c_str());
        pll_msa->sequence[i] = strdup(sequences[i].c_str());
    }

    // std::cout << pll_msa->count << " " << pll_msa->length << std::endl;
    // std::cout << "TESTETSTEST" << std::endl;

    auto& parted_msa = *instance.parted_msa;

    MSA msa(pll_msa);
    instance.opts.use_prob_msa = false;
    parted_msa.full_msa(std::move(msa));
    parted_msa.split_msa();

    assert (opts.use_pattern_compression);
    if (opts.use_pattern_compression)
    {
        LOG_VERB_TS << "Compressing alignment patterns... " << endl;
        bool store_backmap = opts.command == Command::sitelh;
        parted_msa.compress_patterns(store_backmap);

        // temp workaround: since MSA pattern compression calls rand(), it will change all random
        // numbers generated afterwards. so just reset seed to the initial value to ensure that
        // starting trees, BS replicates etc. are the same regardless whether pat.comp is ON or OFF
        srand(opts.random_seed);
    }

    parted_msa.set_model_empirical_params();
    // check_models(instance);

    // load_msa(instance); end
    instance.tip_id_map = instance.parted_msa->taxon_id_map();
    // load_parted_msa(instance); end
#endif

    // check_options(instance);

    instance.random_tree = generate_tree(instance, StartingTree::random, rand());


#if 0
    auto pars_threads = 1;
    build_start_trees(instance, pars_threads);
#else

    string newick_str = tree_str;

    // discard any trailing spaces, newlines etc.
    // stream >> std::ws;

    Tree tree;

    if (!newick_str.empty())
    {
        pll_utree_t * utree = pll_utree_parse_newick_string_unroot(newick_str.c_str());

        libpll_check_error("ERROR reading tree file");

        assert(utree);

        if (utree->edge_count > 2 * utree->tip_count - 3)
        throw runtime_error("Tree contains unifurcations (watch for an extra pair of parentheses)!");

        // tree = Tree(*utree);
        tree.pll_utree(*utree);

        pll_utree_destroy(utree, nullptr);
    }

    tree.reset_tip_ids(instance.tip_id_map);
    prepare_tree(instance, tree);

    instance.start_trees.emplace_back(tree);

#endif


    // opts.num_searches++;
    assert (opts.num_searches == 1);

    instance.bs_converged = false;

    /* run load balancing algorithm */
    balance_load(instance);

}



double thread_infer_ml_compute_llh(RaxmlInstance& instance)
{
    //   auto& worker = instance.get_worker();
    auto const& master_msa = *instance.parted_msa;
    auto const& opts = instance.opts;

    unique_ptr<TreeInfo> treeinfo;

    /* get partitions assigned to the current thread */
    
    // auto const& part_assign = PartitionAssignment();
    auto const& part_assign = instance.proc_part_assign.at(0);

    // if (opts.command == Command::evaluate)
    // {
    //     LOG_INFO << "\nEvaluating " << opts.num_searches <<
    //         " trees" << endl;
    // }
    // else if (instance.run_phase != RaxmlRunPhase::bootstrap)
    // {
    //     LOG_INFO << "\nStarting ML tree search with " << opts.num_searches <<
    //         " distinct starting trees" << endl;
    // }

    // (instance.start_trees.size() > 1 ? LOG_RESULT : LOG_INFO) << endl;

    //   const auto& tree = instance.start_trees.at(start_tree_num-1);
    const auto& tree = instance.start_trees.at(0);
    assert(!tree.empty());
    treeinfo.reset(new TreeInfo(opts, tree, master_msa, instance.tip_msa_idmap, part_assign));

    return treeinfo->loglh();
}


double compute_llh(
    const std::string &tree_str,
    const std::vector<std::string> &labels,
    const std::vector<std::string> &sequences,
    bool tree_rooted
) {
    RaxmlInstance instance;

    build_raxml_instance(instance, tree_str, labels, sequences, tree_rooted);

    auto ret = thread_infer_ml_compute_llh(instance);

    return ret;
}



std::tuple<std::string, double, double> thread_infer_ml_optimize_brlen(RaxmlInstance& instance)
{
    auto const& master_msa = *instance.parted_msa;
    auto const& opts = instance.opts;

    unique_ptr<TreeInfo> treeinfo;

    auto const& part_assign = instance.proc_part_assign.at(0);

    //   const auto& tree = instance.start_trees.at(start_tree_num-1);
    const auto& tree = instance.start_trees.at(0);
    assert(!tree.empty());
    treeinfo.reset(new TreeInfo(opts, tree, master_msa, instance.tip_msa_idmap, part_assign));

    auto loglh_init = treeinfo->loglh();

    // TODO
    // CheckpointManager cm(opts);
    // Optimizer optimizer(opts);
    // optimizer.evaluate(*treeinfo, cm);

    double fast_modopt_eps = 10.0;
    auto loglh_opt = treeinfo->optimize_branches(fast_modopt_eps, 1);


    auto opt_tree = treeinfo->pll_treeinfo().tree;

    auto nodes_count = opt_tree->tip_count + opt_tree->inner_count;

    char *newick = pll_utree_export_newick(opt_tree->nodes[nodes_count-1],NULL);
    std::string out_tree_str(newick);
    free(newick);


    return std::make_tuple(out_tree_str, loglh_init, loglh_opt);
}




std::tuple<std::string, double, double> optimize_brlen(
    const std::string &tree_str,
    const std::vector<std::string> &labels,
    const std::vector<std::string> &sequences,
    bool tree_rooted,
    int iters) {

    RaxmlInstance instance;

    build_raxml_instance(instance, tree_str, labels, sequences, tree_rooted);

    auto ret = thread_infer_ml_optimize_brlen(instance);

    return ret;

}






void test_func() {
    Options o;
}




PYBIND11_MODULE(cpp_binding, m) {
    m.def("compute_llh", &compute_llh, "Compute likelihood for a given tree and sequences");
    m.def("optimize_brlen", &optimize_brlen, "Optimize branch length for a given tree and sequences");
    m.def("test_func", &test_func, "Just a test function");
}
