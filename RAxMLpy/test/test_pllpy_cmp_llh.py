# import pllpy
import raxmlpy

with open("G_l_1000_n_40_0.0_0.09_24.phy", "r") as file:
    first_line = file.readline().strip()
    count, length = map(int, first_line.split())

    # 读取剩余的行
    sequences = []
    for _ in range(count):
        line = file.readline().strip()
        parts = line.split()
        label = parts[0]
        sequence = ' '.join(parts[1:])
        sequences.append((label, sequence))


with open("G_l_1000_n_40_0.0_0.09_24_raw.tre", "r") as file:
    tree_str = file.readline().strip()

msa = {
    "labels": [label for label, _ in sequences],
    "sequences": [sequence for _, sequence in sequences]
}

# P = pllpy.PllPy(msa)

# def test():
#     tree_op, logll, logllop = P.optimize_brlen(tree_str, 3)


# import timeit


# # 设置测试次数，例如测试100次
# number_of_tests = 100

# # 使用timeit.timeit来测量平均运行时间
# average_time = timeit.timeit('test()', setup='from __main__ import test', number=number_of_tests) / number_of_tests

# print(f"平均运行时间: {average_time} 秒")


# print(tree_op)

# tree_op, logll, logllop = pllpy.optimize_brlen(tree_str, msa, is_root=False, iters=3)

logll = raxmlpy.compute_llh(tree_str, msa, is_root=False)


print(logll)

