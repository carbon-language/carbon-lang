#!/usr/bin/env python3

"""Generates C++ header to support LLVM-style RTTI for a class hierarchy.

# Background

A C++ class hierarchy supported by this script consists of *abstract* classes,
which can be inherited from but can't be instantiated, and *concrete* classes,
which can be instantiated but can't be inherited from. Classes can inherit from
at most one other class in the hierarchy; a class that doesn't inherit from
any other class is called a *root* class, and it cannot be concrete.

This script also supports *interface view* classes, which do not participate
in inheritance but are closely tied to a class hierarchy. An interface view
can be thought of as a non-owning pointer to an instance of a class in the
hierarchy that implements some interface. An interface is defined by
duck-typing rather than by inheritance, so interfaces are used to model APIs
that "cut across" the primary inheritance hierarchy.

An interface is not actually a C++ type, but this script treats it as a
class for most purposes. In particular, interfaces notionally participate
in inheritance. A non-interface class cannot inherit from an interface, but
it can *implement* an interface, and the associated interface view can only
point to objects that implement the interface. An interface view class
should be named by appending `View` to the name of the interface.

# Input format

This script's input file declares every class and interface in the hierarchy,
and specifies the inheritance and interface-implementation relationships among
them. A class C can inherit from at most one parent, and may implement any
number of interfaces, subject to the following restrictions:
- If C does not have a parent, it must be declared as a root.
- If C is not an interface, its parent must not be an interface.
- If C is a root or an interface, it cannot implement any interfaces.
- For each interface I that C implements, the nearest non-interface ancestor
  of I must be an ancestor of C.

The input file consists of comment lines starting with `#`, whitespace lines,
and one `;`-terminated line for each class. The core of a line is `class`
followed by the class name. `class` can be prefixed with `root`, `abstract`,
or `interface` to specify the corresponding kind of class; if there is no
prefix, the class is concrete. If the class is not a root class, the name is
followed by `:` and then the name of the class it inherits from. This can
optionally be followed by a comma, and then a comma-separated list of the
interfaces it implements. A class cannot inherit from or implement classes
defined later in the file.

For example:

root class R;
abstract class A : R;
interface class I : R;
abstract class B : R, I;
class C : A;
class D : B;
class E : A, I;

# Output

For each non-concrete class `Foo`, the generated header file will contain
`enum class FooKind`, which has an enumerator for each concrete class derived
from `Foo`, with a name that matches the concrete class name.

For each non-root abstract or concrete class `Foo` whose root class is `Root`,
the generated header file will also contain a function
`bool InheritsFromFoo(RootKind kind)`, which returns true if the value of `kind`
corresponds to a class that is derived from `Foo`. This function can be used to
implement `Foo::classof`.

All enumerators that represent the same concrete class will have the same
numeric value, so you can use `static_cast` to convert between the enum types
for different classes that have a common root, so long as the enumerator value
is present in both types. As a result, `InheritsFromFoo` can be used to
determine whether casting to `FooKind` is safe.

For each interface `Foo`, the generated header file will also contain a
variable template `ImplementsFoo<T>`, which is true if `T` implements `Foo`
or inherits from a class that implements `Foo`, and false otherwise.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import enum
import re
import sys


class Class:
    """Metadata about a class from the input file.

    This consists of information

    Attributes set at construction:
      name: The class name.
      kind: The class kind (root, abstract, interface, or concrete)
      ancestors: A list of Class objects representing the class's ancestors,
        starting with the root and ending with the current class's parent.
      interfaces: A list of Class objects representing the interfaces the class
        implements.
      children: A list of Class objects representing the classes that are
        derived directly from this one.

    Attributes set by Finalize():
      id (CONCRETE only): The class's numeric ID, which will become its
        enumerator value in the generated C++ code.
      id_range (ROOT and ABSTRACT only): A pair such that a Class
        object `c` represents a concrete class derived from `self` if and only
        if c.id >= self.id_range[0] and c.id < self.id_range[1].
      leaf_ids (INTERFACE only): A set containing the IDs of all concrete
        classes derived from this interface.
      leaves (ROOT only): A list of all concrete classes derived from this one,
        indexed by their IDs.
    """

    Kind = enum.Enum("Kind", "ROOT ABSTRACT INTERFACE CONCRETE")

    def __init__(self, name, kind, parent, interfaces):
        self.name = name
        self.kind = kind
        self.interfaces = interfaces

        assert (parent is None) == (kind == Class.Kind.ROOT)
        if parent is None:
            self.ancestors = []
        else:
            self.ancestors = parent.ancestors + [parent]

        if self.kind == Class.Kind.ROOT:
            self.leaves = []
            self.id_range = None
        elif self.kind == Class.Kind.ABSTRACT:
            self.id_range = None
        elif self.kind == Class.Kind.INTERFACE:
            self.leaf_ids = set()
        else:
            self.id = None

        self.children = []

        if parent:
            parent.children.append(self)

        for interface in self.interfaces:
            interface.children.append(self)

    def Parent(self):
        """Returns this Class's parent."""
        return self.ancestors[-1]

    def NearestNonInterfaceAncestor(self):
        """Returns this Class's nearest non-interface ancestor."""
        for ancestor in reversed(self.ancestors):
            if ancestor.kind != Class.Kind.INTERFACE:
                return ancestor

    def Root(self):
        """Returns the root Class of this hierarchy."""
        if self.kind == Class.Kind.ROOT:
            return self
        else:
            return self.ancestors[0]

    def _RegisterLeaf(self, leaf):
        """Records that `leaf` is derived from self.

        Also recursively updates the parent and interfaces of self. leaf.id must
        already be populated, and leaves must be registered in order of ID. This
        operation is idempotent."""
        already_visited = False
        if self.kind == Class.Kind.ROOT:
            if leaf.id == len(self.leaves):
                self.leaves.append(leaf)
            else:
                assert leaf.id + 1 == len(self.leaves)
                assert self.leaves[leaf.id] == leaf
                already_visited = True
        if self.kind in [Class.Kind.ROOT, Class.Kind.ABSTRACT]:
            if self not in leaf.ancestors:
                sys.exit(
                    f"{leaf.name} derived from {self.name}, but has a"
                    + " different root"
                )
            if not self.id_range:
                self.id_range = (leaf.id, leaf.id + 1)
            elif self.id_range[1] == leaf.id:
                self.id_range = (self.id_range[0], self.id_range[1] + 1)
            else:
                assert self.id_range[1] == leaf.id + 1
                already_visited = True

        elif self.kind == Class.Kind.INTERFACE:
            if leaf.id in self.leaf_ids:
                already_visited = True
            else:
                self.leaf_ids.add(leaf.id)

        if not already_visited:
            if self.kind != Class.Kind.ROOT:
                self.Parent()._RegisterLeaf(leaf)
            for interface in self.interfaces:
                interface._RegisterLeaf(leaf)

    def Finalize(self):
        """Populates additional attributes for `self` and derived Classes.

        Each Class can only be finalized once, after which no additional Classes
        can be derived from it.
        """
        if self.kind == Class.Kind.CONCRETE:
            self.id = len(self.Root().leaves)
            self._RegisterLeaf(self)
        elif self.kind in [Class.Kind.ROOT, Class.Kind.ABSTRACT]:
            for child in self.children:
                child.Finalize()


_LINE_PATTERN = r"""(?P<prefix> \w*) \s*
                 class \s+
                 (?P<name> \w+)
                 (?: \s*:\s* (?P<parent> \w+)
                   (?: , (?P<interfaces> .*) )?
                 )?
                 ;$"""


def main():
    input_filename = sys.argv[1]
    with open(input_filename) as file:
        lines = file.readlines()

    classes = dict()
    for line_num, line in enumerate(lines, 1):
        if line.startswith("#") or line.strip() == "":
            continue
        match_result = re.match(_LINE_PATTERN, line.strip(), re.VERBOSE)
        if not match_result:
            sys.exit(f"Invalid format on line {line_num}")

        prefix = match_result.group("prefix")
        if prefix == "":
            kind = Class.Kind.CONCRETE
        elif prefix == "root":
            kind = Class.Kind.ROOT
        elif prefix == "abstract":
            kind = Class.Kind.ABSTRACT
        elif prefix == "interface":
            kind = Class.Kind.INTERFACE
        else:
            sys.exit(f"Unrecognized class prefix '{prefix}' on line {line_num}")

        parent = None
        if match_result.group("parent"):
            if kind == Class.Kind.ROOT:
                sys.exit(f"Root class cannot have parent on line {line_num}")
            parent_name = match_result.group("parent")
            parent = classes[parent_name]
            if not parent:
                sys.exit(f"Unknown class '{parent_name}' on line {line_num}")
            if parent.kind == Class.Kind.CONCRETE:
                sys.exit(f"{parent.name} cannot be a parent on line {line_num}")
            elif parent.kind == Class.Kind.INTERFACE:
                if kind != Class.Kind.INTERFACE:
                    sys.exit(
                        "Interface cannot be parent of non-interface on"
                        + f" line {line_num}"
                    )
        else:
            if kind != Class.Kind.ROOT:
                sys.exit(
                    f"Non-root class must have a parent on line {line_num}"
                )

        interfaces = []
        if match_result.group("interfaces"):
            if kind == Class.Kind.INTERFACE:
                sys.exit(
                    f"Interface cannot implement interfaces on line {line_num}"
                )
            for unstripped_name in match_result.group("interfaces").split(","):
                interface_name = unstripped_name.strip()
                interface = classes[interface_name]
                if not interface:
                    sys.exit(
                        f"Unknown class '{interface_name}' on line {line_num}"
                    )
                if interface.kind != Class.Kind.INTERFACE:
                    sys.exit(
                        f"'{interface_name}' used as interface on"
                        + f" line {line_num}"
                    )
                interface_ancestor = interface.NearestNonInterfaceAncestor()
                if interface_ancestor not in parent.ancestors + [parent]:
                    sys.exit(
                        f"Class that implements {interface_name} must"
                        + f" inherit from {interface_ancestor.name}"
                        + f" on line {line_num}"
                    )

                interfaces.append(interface)

        classes[match_result.group("name")] = Class(
            match_result.group("name"), kind, parent, interfaces
        )

    for node in classes.values():
        if node.kind == Class.Kind.ROOT:
            node.Finalize()

    print(
        f"// Generated from {input_filename} by"
        + " executable_semantics/gen_rtti.py\n"
    )
    guard_macro = (
        input_filename.upper().translate(str.maketrans({"/": "_", ".": "_"}))
        + "_"
    )
    print(f"#ifndef {guard_macro}")
    print(f"#define {guard_macro}")
    print("\nnamespace Carbon {\n")

    for node in classes.values():
        if node.kind != Class.Kind.CONCRETE:
            if node.kind == Class.Kind.INTERFACE:
                ids = sorted(node.leaf_ids)
            else:
                ids = range(node.id_range[0], node.id_range[1])
            print(f"enum class {node.name}Kind {{")
            for id in ids:
                print(f"  {node.Root().leaves[id].name} = {id},")
            print("};\n")

        if node.kind in [Class.Kind.ABSTRACT, Class.Kind.CONCRETE]:
            print(f"class {node.name};\n")
            print(
                f"inline bool InheritsFrom{node.name}({node.Root().name}Kind"
                + " kind) {"
            )
            if node.kind == Class.Kind.ABSTRACT:
                if node.id_range[0] == node.id_range[1]:
                    print("  return false;")
                else:
                    range_begin = node.Root().leaves[node.id_range[0]].name
                    print(
                        f"  return kind >= {node.Root().name}Kind"
                        + f"::{range_begin}"
                    )
                    if node.id_range[1] < len(node.Root().leaves):
                        range_end = node.Root().leaves[node.id_range[1]].name
                        print(
                            f"      && kind < {node.Root().name}Kind"
                            + f"::{range_end}"
                        )
                    print("      ;")
            elif node.kind == Class.Kind.CONCRETE:
                print(
                    f"    return kind == {node.Root().name}Kind::{node.name};"
                )
            print("}\n")

    for node in classes.values():
        if node.kind == Class.Kind.INTERFACE:
            print("template <typename NodeType>")
            print(f"constexpr bool Implements{node.name} = false;")

            queue = node.children.copy()
            while queue:
                descendant = queue.pop(0)
                queue.extend(descendant.children)
                print(
                    "template <> inline constexpr bool "
                    + f"Implements{node.name}<{descendant.name}> = true;"
                )

    print("}  // namespace Carbon\n")
    print(f"#endif  // {guard_macro}")


if __name__ == "__main__":
    main()
