import pprint
import os

import componentinfo

class LLVMProjectInfo(object):
    @staticmethod
    def load_infos_from_path(llvmbuild_source_root):
        # FIXME: Implement a simple subpath file list cache, so we don't restat
        # directories we have already traversed.

        # First, discover all the LLVMBuild.txt files.
        for dirpath,dirnames,filenames in os.walk(llvmbuild_source_root,
                                                  followlinks = True):
            # If there is no LLVMBuild.txt file in a directory, we don't recurse
            # past it. This is a simple way to prune our search, although it
            # makes it easy for users to add LLVMBuild.txt files in places they
            # won't be seen.
            if 'LLVMBuild.txt' not in filenames:
                del dirnames[:]
                continue

            # Otherwise, load the LLVMBuild file in this directory.
            assert dirpath.startswith(llvmbuild_source_root)
            subpath = '/' + dirpath[len(llvmbuild_source_root)+1:]
            llvmbuild_path = os.path.join(dirpath, 'LLVMBuild.txt')
            for info in componentinfo.load_from_path(llvmbuild_path, subpath):
                yield info

    @staticmethod
    def load_from_path(source_root, llvmbuild_source_root):
        infos = list(
            LLVMProjectInfo.load_infos_from_path(llvmbuild_source_root))

        return LLVMProjectInfo(source_root, infos)

    def __init__(self, source_root, component_infos):
        self.source_root = source_root
        self.component_infos = component_infos

def main():
    from optparse import OptionParser, OptionGroup
    parser = OptionParser("usage: %prog [options]")
    parser.add_option("", "--source-root", dest="source_root", metavar="PATH",
                      help="Path to the LLVM source (inferred if not given)",
                      action="store", default=None)
    parser.add_option(
        "", "--llvmbuild-source-root", dest="llvmbuild_source_root",
        help="If given, an alternate path to search for LLVMBuild.txt files",
        action="store", default=None, metavar="PATH")
    (opts, args) = parser.parse_args()

    # Determine the LLVM source path, if not given.
    source_root = opts.source_root
    if source_root:
        if not os.path.exists(os.path.join(source_root, 'lib', 'VMCore',
                                           'Function.cpp')):
            parser.error('invalid LLVM source root: %r' % source_root)
    else:
        llvmbuild_path = os.path.dirname(__file__)
        llvm_build_path = os.path.dirname(llvmbuild_path)
        utils_path = os.path.dirname(llvm_build_path)
        source_root = os.path.dirname(utils_path)
        if not os.path.exists(os.path.join(source_root, 'lib', 'VMCore',
                                           'Function.cpp')):
            parser.error('unable to infer LLVM source root, please specify')

    # Construct the LLVM project information.
    llvmbuild_source_root = opts.llvmbuild_source_root or source_root
    project_info = LLVMProjectInfo.load_from_path(
        source_root, llvmbuild_source_root)

if __name__=='__main__':
    main()
