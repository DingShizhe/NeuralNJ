from ete3 import Tree

def calculate_rf_distance(file1, file2):
    t1=Tree(file1)
    # print(os.path.join(preds, tree.split('.tre')[0]+'.pf.nwk'))
    t2=Tree(file2)
    norm_rf_dist = t1.compare(t2,unrooted=True)['norm_rf']
    rf = t1.compare(t2,unrooted=True)['rf']
    return rf, norm_rf_dist

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reftree", type=str, required=True)
    parser.add_argument("--inftree", type=str, required=True)
    args = parser.parse_args()
    rf, norm_rf_dist = calculate_rf_distance(args.reftree, args.inftree)
    print(f"RF distance: {rf}, Normalized RF distance: {norm_rf_dist}")
