"""
Descriptor objects for entities that are part of the LLVM project.
"""

import ConfigParser
import sys

class ComponentInfo(object):
    """
    Base class for component descriptions.
    """

    type_name = None

    def __init__(self, subpath, name, dependencies, parent):
        if not subpath.startswith('/'):
            raise ValueError,"invalid subpath: %r" % subpath
        self.subpath = subpath
        self.name = name
        self.dependencies = list(dependencies)

        # The name of the parent component to logically group this component
        # under.
        self.parent = parent

class GroupComponentInfo(ComponentInfo):
    """
    Group components have no semantics as far as the build system are concerned,
    but exist to help organize other components into a logical tree structure.
    """

    type_name = 'Group'

    def __init__(self, subpath, name, parent):
        ComponentInfo.__init__(self, subpath, name, [], parent)

class LibraryComponentInfo(ComponentInfo):
    type_name = 'Library'

    def __init__(self, subpath, name, dependencies, parent, library_name = None,
                 required_libraries = [], add_to_library_groups = []):
        ComponentInfo.__init__(self, subpath, name, dependencies, parent)

        # If given, the name to use for the library instead of deriving it from
        # the component name.
        self.library_name = library_name

        # The names of the library components which are required when linking
        # with this component.
        self.required_libraries = list(required_libraries)

        # The names of the library group components this component should be
        # considered part of.
        self.add_to_library_groups = list(add_to_library_groups)

class LibraryGroupComponentInfo(ComponentInfo):
    type_name = 'LibraryGroup'

    def __init__(self, subpath, name, parent, required_libraries = [],
                 add_to_library_groups = []):
        ComponentInfo.__init__(self, subpath, name, [], parent)

        # The names of the library components which are required when linking
        # with this component.
        self.required_libraries = list(required_libraries)

        # The names of the library group components this component should be
        # considered part of.
        self.add_to_library_groups = list(add_to_library_groups)

class ToolComponentInfo(ComponentInfo):
    type_name = 'Tool'

    def __init__(self, subpath, name, dependencies, parent,
                 required_libraries = []):
        ComponentInfo.__init__(self, subpath, name, dependencies, parent)

        # The names of the library components which are required to link this
        # tool.
        self.required_libraries = list(required_libraries)

class BuildToolComponentInfo(ToolComponentInfo):
    type_name = 'BuildTool'

_component_type_map = dict(
    (t.type_name, t)
    for t in (GroupComponentInfo,
              LibraryComponentInfo, LibraryGroupComponentInfo,
              ToolComponentInfo, BuildToolComponentInfo))
def load_from_path(path, subpath):
    # Load the LLVMBuild.txt file as an .ini format file.
    parser = ConfigParser.RawConfigParser()
    parser.read(path)

    # We load each section which starts with 'component' as a distinct component
    # description (so multiple components can be described in one file).
    for section in parser.sections():
        if not section.startswith('component'):
            # We don't expect arbitrary sections currently, warn the user.
            print >>sys.stderr, "warning: ignoring unknown section %r in %r" % (
                section, path)
            continue

        # Load the component that this section describes. For now we just do
        # this the trivial way by letting python validate the argument
        # assignment. This is simple, but means users see lame diagnostics. We
        # should audit the component manually, eventually.
        if not parser.has_option(section, 'type'):
            print >>sys.stderr, "error: invalid component %r in %r: %s" % (
                section, path, "no component type")
            raise SystemExit, 1

        type_name = parser.get(section, 'type')
        type_class = _component_type_map.get(type_name)
        if type_class is None:
            print >>sys.stderr, "error: invalid component %r in %r: %s" % (
                section, path, "invalid component type: %r" % type_name)
            raise SystemExit, 1

        items = dict(parser.items(section))
        items['subpath'] = subpath
        items.pop('type')

        # Instantiate the component based on the remaining values.
        try:
            info = type_class(**items)
        except TypeError:
            print >>sys.stderr, "error: invalid component %r in %r: %s" % (
                section, path, "unable to instantiate: %r" % type_name)
            import traceback
            traceback.print_exc()
            raise SystemExit, 1

        yield info
