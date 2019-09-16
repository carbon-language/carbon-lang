import difflib
import functools
import getopt
import os
import sys

class DiffFlags():
    def __init__(self):
        self.ignore_all_space = False
        self.ignore_space_change = False
        self.unified_diff = False
        self.recursive_diff = False
        self.strip_trailing_cr = False

def getDirTree(path, basedir=""):
    # Tree is a tuple of form (dirname, child_trees).
    # An empty dir has child_trees = [], a file has child_trees = None.
    child_trees = []
    for dirname, child_dirs, files in os.walk(os.path.join(basedir, path)):
        for child_dir in child_dirs:
            child_trees.append(getDirTree(child_dir, dirname))
        for filename in files:
            child_trees.append((filename, None))
        return path, sorted(child_trees)

def compareTwoFiles(flags, filepaths):
    compare_bytes = False
    encoding = None
    filelines = []
    for file in filepaths:
        try:
            with open(file, 'r') as f:
                filelines.append(f.readlines())
        except UnicodeDecodeError:
            try:
                with io.open(file, 'r', encoding="utf-8") as f:
                    filelines.append(f.readlines())
                encoding = "utf-8"
            except:
                compare_bytes = True

    if compare_bytes:
        return compareTwoBinaryFiles(flags, filepaths)
    else:
        return compareTwoTextFiles(flags, filepaths, encoding)

def compareTwoBinaryFiles(flags, filepaths):
    filelines = []
    for file in filepaths:
        with open(file, 'rb') as f:
            filelines.append(f.readlines())

    exitCode = 0
    if hasattr(difflib, 'diff_bytes'):
        # python 3.5 or newer
        diffs = difflib.diff_bytes(difflib.unified_diff, filelines[0], filelines[1], filepaths[0].encode(), filepaths[1].encode())
        diffs = [diff.decode() for diff in diffs]
    else:
        # python 2.7
        if flags.unified_diff:
            func = difflib.unified_diff
        else:
            func = difflib.context_diff
        diffs = func(filelines[0], filelines[1], filepaths[0], filepaths[1])

    for diff in diffs:
        sys.stdout.write(diff)
        exitCode = 1
    return exitCode

def compareTwoTextFiles(flags, filepaths, encoding):
    filelines = []
    for file in filepaths:
        if encoding is None:
            with open(file, 'r') as f:
                filelines.append(f.readlines())
        else:
            with io.open(file, 'r', encoding=encoding) as f:
                filelines.append(f.readlines())

    exitCode = 0
    def compose2(f, g):
        return lambda x: f(g(x))

    f = lambda x: x
    if flags.strip_trailing_cr:
        f = compose2(lambda line: line.rstrip('\r'), f)
    if flags.ignore_all_space or flags.ignore_space_change:
        ignoreSpace = lambda line, separator: separator.join(line.split())
        ignoreAllSpaceOrSpaceChange = functools.partial(ignoreSpace, separator='' if flags.ignore_all_space else ' ')
        f = compose2(ignoreAllSpaceOrSpaceChange, f)

    for idx, lines in enumerate(filelines):
        filelines[idx]= [f(line) for line in lines]

    func = difflib.unified_diff if flags.unified_diff else difflib.context_diff
    for diff in func(filelines[0], filelines[1], filepaths[0], filepaths[1]):
        sys.stdout.write(diff)
        exitCode = 1
    return exitCode

def printDirVsFile(dir_path, file_path):
    if os.path.getsize(file_path):
        msg = "File %s is a directory while file %s is a regular file"
    else:
        msg = "File %s is a directory while file %s is a regular empty file"
    sys.stdout.write(msg % (dir_path, file_path) + "\n")

def printFileVsDir(file_path, dir_path):
    if os.path.getsize(file_path):
        msg = "File %s is a regular file while file %s is a directory"
    else:
        msg = "File %s is a regular empty file while file %s is a directory"
    sys.stdout.write(msg % (file_path, dir_path) + "\n")

def printOnlyIn(basedir, path, name):
    sys.stdout.write("Only in %s: %s\n" % (os.path.join(basedir, path), name))

def compareDirTrees(flags, dir_trees, base_paths=["", ""]):
    # Dirnames of the trees are not checked, it's caller's responsibility,
    # as top-level dirnames are always different. Base paths are important
    # for doing os.walk, but we don't put it into tree's dirname in order
    # to speed up string comparison below and while sorting in getDirTree.
    left_tree, right_tree = dir_trees[0], dir_trees[1]
    left_base, right_base = base_paths[0], base_paths[1]

    # Compare two files or report file vs. directory mismatch.
    if left_tree[1] is None and right_tree[1] is None:
        return compareTwoFiles(flags,
                               [os.path.join(left_base, left_tree[0]),
                                os.path.join(right_base, right_tree[0])])

    if left_tree[1] is None and right_tree[1] is not None:
        printFileVsDir(os.path.join(left_base, left_tree[0]),
                       os.path.join(right_base, right_tree[0]))
        return 1

    if left_tree[1] is not None and right_tree[1] is None:
        printDirVsFile(os.path.join(left_base, left_tree[0]),
                       os.path.join(right_base, right_tree[0]))
        return 1

    # Compare two directories via recursive use of compareDirTrees.
    exitCode = 0
    left_names = [node[0] for node in left_tree[1]]
    right_names = [node[0] for node in right_tree[1]]
    l, r = 0, 0
    while l < len(left_names) and r < len(right_names):
        # Names are sorted in getDirTree, rely on that order.
        if left_names[l] < right_names[r]:
            exitCode = 1
            printOnlyIn(left_base, left_tree[0], left_names[l])
            l += 1
        elif left_names[l] > right_names[r]:
            exitCode = 1
            printOnlyIn(right_base, right_tree[0], right_names[r])
            r += 1
        else:
            exitCode |= compareDirTrees(flags,
                                        [left_tree[1][l], right_tree[1][r]],
                                        [os.path.join(left_base, left_tree[0]),
                                        os.path.join(right_base, right_tree[0])])
            l += 1
            r += 1

    # At least one of the trees has ended. Report names from the other tree.
    while l < len(left_names):
        exitCode = 1
        printOnlyIn(left_base, left_tree[0], left_names[l])
        l += 1
    while r < len(right_names):
        exitCode = 1
        printOnlyIn(right_base, right_tree[0], right_names[r])
        r += 1
    return exitCode

def main(argv):
    args = argv[1:]
    try:
        opts, args = getopt.gnu_getopt(args, "wbur", ["strip-trailing-cr"])
    except getopt.GetoptError as err:
        sys.stderr.write("Unsupported: 'diff': %s\n" % str(err))
        sys.exit(1)

    flags = DiffFlags()
    filelines, filepaths, dir_trees = ([] for i in range(3))
    for o, a in opts:
        if o == "-w":
            flags.ignore_all_space = True
        elif o == "-b":
            flags.ignore_space_change = True
        elif o == "-u":
            flags.unified_diff = True
        elif o == "-r":
            flags.recursive_diff = True
        elif o == "--strip-trailing-cr":
            flags.strip_trailing_cr = True
        else:
            assert False, "unhandled option"

    if len(args) != 2:
        sys.stderr.write("Error: missing or extra operand\n")
        sys.exit(1)

    exitCode = 0
    try:
        for file in args:
            if not os.path.isabs(file):
                file = os.path.realpath(os.path.join(os.getcwd(), file))

            if flags.recursive_diff:
                dir_trees.append(getDirTree(file))
            else:
                filepaths.append(file)

        if not flags.recursive_diff:
            exitCode = compareTwoFiles(flags, filepaths)
        else:
            exitCode = compareDirTrees(flags, dir_trees)

    except IOError as err:
        sys.stderr.write("Error: 'diff' command failed, %s\n" % str(err))
        exitCode = 1

    sys.exit(exitCode)

if __name__ == "__main__":
    main(sys.argv)
