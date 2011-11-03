"""
Descriptor objects for entities that are part of the LLVM project.
"""

import ConfigParser
import sys

from util import *

class ParseError(Exception):
    pass

class ComponentInfo(object):
    """
    Base class for component descriptions.
    """

    type_name = None

    @staticmethod
    def parse_items(items, has_dependencies = True):
        kwargs = {}
        kwargs['name'] = items.get_string('name')
        kwargs['parent'] = items.get_optional_string('parent')
        if has_dependencies:
            kwargs['dependencies'] = items.get_list('dependencies')
        return kwargs

    def __init__(self, subpath, name, dependencies, parent):
        if not subpath.startswith('/'):
            raise ValueError,"invalid subpath: %r" % subpath
        self.subpath = subpath
        self.name = name
        self.dependencies = list(dependencies)

        # The name of the parent component to logically group this component
        # under.
        self.parent = parent

    def get_component_references(self):
        """get_component_references() -> iter

        Return an iterator over the named references to other components from
        this object. Items are of the form (reference-type, component-name).
        """

        # Parent references are handled specially.
        for r in self.dependencies:
            yield ('dependency', r)

class GroupComponentInfo(ComponentInfo):
    """
    Group components have no semantics as far as the build system are concerned,
    but exist to help organize other components into a logical tree structure.
    """

    type_name = 'Group'

    @staticmethod
    def parse(subpath, items):
        kwargs = ComponentInfo.parse_items(items, has_dependencies = False)
        return GroupComponentInfo(subpath, **kwargs)

    def __init__(self, subpath, name, parent):
        ComponentInfo.__init__(self, subpath, name, [], parent)

class LibraryComponentInfo(ComponentInfo):
    type_name = 'Library'

    @staticmethod
    def parse(subpath, items):
        kwargs = ComponentInfo.parse_items(items)
        kwargs['library_name'] = items.get_optional_string('name')
        kwargs['required_libraries'] = items.get_list('required_libraries')
        kwargs['add_to_library_groups'] = items.get_list(
            'add_to_library_groups')
        return LibraryComponentInfo(subpath, **kwargs)

    def __init__(self, subpath, name, dependencies, parent, library_name,
                 required_libraries, add_to_library_groups):
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

    def get_component_references(self):
        for r in ComponentInfo.get_component_references(self):
            yield r
        for r in self.required_libraries:
            yield ('required library', r)
        for r in self.add_to_library_groups:
            yield ('library group', r)

class LibraryGroupComponentInfo(ComponentInfo):
    type_name = 'LibraryGroup'

    @staticmethod
    def parse(subpath, items):
        kwargs = ComponentInfo.parse_items(items, has_dependencies = False)
        kwargs['required_libraries'] = items.get_list('required_libraries')
        kwargs['add_to_library_groups'] = items.get_list(
            'add_to_library_groups')
        return LibraryGroupComponentInfo(subpath, **kwargs)

    def __init__(self, subpath, name, parent, required_libraries = [],
                 add_to_library_groups = []):
        ComponentInfo.__init__(self, subpath, name, [], parent)

        # The names of the library components which are required when linking
        # with this component.
        self.required_libraries = list(required_libraries)

        # The names of the library group components this component should be
        # considered part of.
        self.add_to_library_groups = list(add_to_library_groups)

    def get_component_references(self):
        for r in ComponentInfo.get_component_references(self):
            yield r
        for r in self.required_libraries:
            yield ('required library', r)
        for r in self.add_to_library_groups:
            yield ('library group', r)

class ToolComponentInfo(ComponentInfo):
    type_name = 'Tool'

    @staticmethod
    def parse(subpath, items):
        kwargs = ComponentInfo.parse_items(items)
        kwargs['required_libraries'] = items.get_list('required_libraries')
        return ToolComponentInfo(subpath, **kwargs)

    def __init__(self, subpath, name, dependencies, parent,
                 required_libraries):
        ComponentInfo.__init__(self, subpath, name, dependencies, parent)

        # The names of the library components which are required to link this
        # tool.
        self.required_libraries = list(required_libraries)

    def get_component_references(self):
        for r in ComponentInfo.get_component_references(self):
            yield r
        for r in self.required_libraries:
            yield ('required library', r)

class BuildToolComponentInfo(ToolComponentInfo):
    type_name = 'BuildTool'

    @staticmethod
    def parse(subpath, items):
        kwargs = ComponentInfo.parse_items(items)
        kwargs['required_libraries'] = items.get_list('required_libraries')
        return BuildToolComponentInfo(subpath, **kwargs)

###

class IniFormatParser(dict):
    def get_list(self, key):
        # Check if the value is defined.
        value = self.get(key)
        if value is None:
            return []

        # Lists are just whitespace separated strings.
        return value.split()

    def get_optional_string(self, key):
        value = self.get_list(key)
        if not value:
            return None
        if len(value) > 1:
            raise ParseError("multiple values for scalar key: %r" % key)
        return value[0]

    def get_string(self, key):
        value = self.get_optional_string(key)
        if not value:
            raise ParseError("missing value for required string: %r" % key)
        return value

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
            warning("ignoring unknown section %r in %r" % (section, path))
            continue

        # Determine the type of the component to instantiate.
        if not parser.has_option(section, 'type'):
            fatal("invalid component %r in %r: %s" % (
                    section, path, "no component type"))

        type_name = parser.get(section, 'type')
        type_class = _component_type_map.get(type_name)
        if type_class is None:
            fatal("invalid component %r in %r: %s" % (
                    section, path, "invalid component type: %r" % type_name))

        # Instantiate the component based on the remaining values.
        try:
            info = type_class.parse(subpath,
                                    IniFormatParser(parser.items(section)))
        except TypeError:
            print >>sys.stderr, "error: invalid component %r in %r: %s" % (
                section, path, "unable to instantiate: %r" % type_name)
            import traceback
            traceback.print_exc()
            raise SystemExit, 1
        except ParseError,e:
            fatal("unable to load component %r in %r: %s" % (
                    section, path, e.message))

        yield info
