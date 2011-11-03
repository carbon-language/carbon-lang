import os
import sys

import componentinfo

from util import *

###

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
        # Store our simple ivars.
        self.source_root = source_root
        self.component_infos = component_infos

        # Create the component info map and validate that component names are
        # unique.
        self.component_info_map = {}
        for ci in component_infos:
            existing = self.component_info_map.get(ci.name)
            if existing is not None:
                # We found a duplicate component name, report it and error out.
                fatal("found duplicate component %r (at %r and %r)" % (
                        ci.name, ci.subpath, existing.subpath))
            self.component_info_map[ci.name] = ci

        # Add the root component, which does not go in any list.
        if '$ROOT' in self.component_info_map:
            fatal("project is not allowed to define $ROOT component")
        self.component_info_map['$ROOT'] = componentinfo.GroupComponentInfo(
            '/', '$ROOT', None)

        # Topologically order the component information according to their
        # component references.
        def visit_component_info(ci, current_stack, current_set):
            # Check for a cycles.
            if ci in current_set:
                # We found a cycle, report it and error out.
                cycle_description = ' -> '.join(
                    '%r (%s)' % (ci.name, relation)
                    for relation,ci in current_stack)
                fatal("found cycle to %r after following: %s -> %s" % (
                        ci.name, cycle_description, ci.name))

            # If we have already visited this item, we are done.
            if ci not in components_to_visit:
                return

            # Otherwise, mark the component info as visited and traverse.
            components_to_visit.remove(ci)

            # Validate the parent reference, which we treat specially.
            parent = self.component_info_map.get(ci.parent)
            if parent is None:
                fatal("component %r has invalid reference %r (via %r)" % (
                        ci.name, ci.parent, 'parent'))
            ci.set_parent_instance(parent)

            for relation,referent_name in ci.get_component_references():
                # Validate that the reference is ok.
                referent = self.component_info_map.get(referent_name)
                if referent is None:
                    fatal("component %r has invalid reference %r (via %r)" % (
                            ci.name, referent_name, relation))

                # Visit the reference.
                current_stack.append((relation,ci))
                current_set.add(ci)
                visit_component_info(referent, current_stack, current_set)
                current_set.remove(ci)
                current_stack.pop()

            # Finally, add the component info to the ordered list.
            self.ordered_component_infos.append(ci)

        # FIXME: We aren't actually correctly checking for cycles along the
        # parent edges. Haven't decided how I want to handle this -- I thought
        # about only checking cycles by relation type. If we do that, it falls
        # out easily. If we don't, we should special case the check.

        self.ordered_component_infos = []
        components_to_visit = set(component_infos)
        while components_to_visit:
            visit_component_info(iter(components_to_visit).next(), [], set())

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
