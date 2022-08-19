#!/usr/bin/env python3

"""Generates C++ header to support LLVM-style RTTI for a class hierarchy.

This script should be run through the //explorer:gen_rtti build target.

# Background

A C++ class hierarchy supported by this script consists of *abstract* classes,
which can be inherited from but can't be instantiated, and *concrete* classes,
which can be instantiated but can't be inherited from. Classes can inherit from
at most one other class in the hierarchy; a class that doesn't inherit from
any other class is called a *root* class, and it cannot be concrete.

# Input format

This script's input file declares every class in the hierarchy, and specifies
the parent of each non-root class. The input file consists of comment lines
starting with `#`, whitespace lines, and one `;`-terminated line for each class.
The core of a line is `class` followed by the class name. `class` can be
prefixed with `root` or `abstract` to specify the corresponding kind of class;
if there is no prefix, the class is concrete. If the class is not a root class,
the name is followed by `:` and then the name of the class it inherits from. A
class cannot inherit from a class defined later in the file.

For example:

root class R;
abstract class A : R;
abstract class B : R;
class C : A;
class D : B;
class E : A;

# Output

For each abstract class `Foo`, the generated header file will contain
`enum class FooKind`, which has an enumerator for each concrete class derived
from `Foo`, with a name that matches the concrete class name.

For each non-root class `Foo` whose root class is `Root`, the generated header
file will also contain a function `bool InheritsFromFoo(RootKind kind)`, which
returns true if the value of `kind` corresponds to a class that is derived from
`Foo`. This function can be used to implement `Foo::classof`.

All enumerators that represent the same concrete class will have the same
numeric value, so you can use `static_cast` to convert between the enum types
for different classes that have a common root, so long as the enumerator value
is present in both types. As a result, `InheritsFromFoo` can be used to
determine whether casting to `FooKind` is safe.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import enum
import re
import sys
from typing import Dict, List, Optional, Tuple


class Class:
    """Metadata about a class from the input file.

    This consists of information

    Attributes set at construction:
      name: The class name.
      kind: The class kind (root, abstract, or concrete)
      ancestors: A list of Class objects representing the class's ancestors,
        starting with the root and ending with the current class's parent.
      _children: A list of Class objects representing the classes that are
        derived directly from this one.

    Attributes set by Finalize():
      id (CONCRETE only): The class's numeric ID, which will become its
        enumerator value in the generated C++ code.
      id_range (ROOT and ABSTRACT only): A pair such that a Class
        object `c` represents a concrete class derived from `self` if and only
        if c.id >= self.id_range[0] and c.id < self.id_range[1].
      leaves (ROOT only): A list of all concrete classes derived from this one,
        indexed by their IDs.
    """

    class Kind(enum.Enum):
        ROOT = enum.auto()
        ABSTRACT = enum.auto()
        CONCRETE = enum.auto()

    def __init__(
        self, name: str, kind: Kind, parent: Optional["Class"]
    ) -> None:
        self.name = name
        self.kind = kind

        assert (parent is None) == (kind == Class.Kind.ROOT)
        self.ancestors: List[Class] = []
        if parent is not None:
            self.ancestors = parent.ancestors + [parent]

        if self.kind == Class.Kind.CONCRETE:
            self.id: Optional[int] = None
        else:
            self.id_range: Optional[Tuple[int, int]] = None
            if self.kind == Class.Kind.ROOT:
                self.leaves: List[Class] = []

        if self.kind != Class.Kind.CONCRETE:
            self._children: List[Class] = []

        if parent:
            parent._children.append(self)

    def Parent(self) -> "Class":
        """Returns this Class's parent."""
        return self.ancestors[-1]

    def Root(self) -> "Class":
        """Returns the root Class of this hierarchy."""
        if self.kind == Class.Kind.ROOT:
            return self
        else:
            return self.ancestors[0]

    def _RegisterLeaf(self, leaf: "Class") -> None:
        """Records that `leaf` is derived from self.

        Also recursively updates the parent of self. leaf.id must already be
        populated, and leaves must be registered in order of ID. This operation
        is idempotent.
        """
        already_visited = False
        assert leaf.id is not None
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

        if not already_visited:
            if self.kind != Class.Kind.ROOT:
                self.Parent()._RegisterLeaf(leaf)

    def Finalize(self) -> None:
        """Populates additional attributes for `self` and derived Classes.

        Each Class can only be finalized once, after which no additional Classes
        can be derived from it.
        """
        if self.kind == Class.Kind.CONCRETE:
            self.id = len(self.Root().leaves)
            self._RegisterLeaf(self)
        else:
            for child in self._children:
                child.Finalize()


_LINE_PATTERN = r"""(?P<prefix> \w*) \s*
                 class \s+
                 (?P<name> \w+)
                 (?: \s*:\s* (?P<parent> \w+)
                 )?
                 ;$"""


def main() -> None:
    input_filename = sys.argv[1]
    header_filename = sys.argv[2]
    cpp_filename = sys.argv[3]
    relative_header = sys.argv[4]

    with open(input_filename) as file:
        lines = file.readlines()

    classes: Dict[str, Class] = {}
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
        else:
            if kind != Class.Kind.ROOT:
                sys.exit(
                    f"Non-root class must have a parent on line {line_num}"
                )

        classes[match_result.group("name")] = Class(
            match_result.group("name"), kind, parent
        )

    for node in classes.values():
        if node.kind == Class.Kind.ROOT:
            node.Finalize()

    header_file = open(header_filename, "w")
    sys.stdout = header_file

    print(f"// Generated from {input_filename} by explorer/gen_rtti.py\n")
    trans_table = str.maketrans({"/": "_", ".": "_"})
    guard_macro = input_filename.upper().translate(trans_table) + "_"
    print(f"#ifndef {guard_macro}")
    print(f"#define {guard_macro}")
    print("\n#include <string_view>")
    print("\nnamespace Carbon {\n")

    for node in classes.values():
        if node.kind != Class.Kind.CONCRETE:
            assert node.id_range is not None
            ids = range(node.id_range[0], node.id_range[1])
            print(f"enum class {node.name}Kind {{")
            for id in ids:
                print(f"  {node.Root().leaves[id].name} = {id},")
            print("};\n")

            print(f"std::string_view {node.name}KindName({node.name}Kind k);\n")

        if node.kind in [Class.Kind.ABSTRACT, Class.Kind.CONCRETE]:
            print(
                f"inline bool InheritsFrom{node.name}({node.Root().name}Kind"
                + " kind) {"
            )
            if node.kind == Class.Kind.ABSTRACT:
                assert node.id_range is not None
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

    print("}  // namespace Carbon\n")
    print(f"#endif  // {guard_macro}")

    header_file.close()

    cpp_file = open(cpp_filename, "w")
    sys.stdout = cpp_file

    print(f"// Generated from {input_filename} by explorer/gen_rtti.py\n")
    print(f'#include "{relative_header}"')
    print("\nnamespace Carbon {\n")
    for node in classes.values():
        if node.kind != Class.Kind.CONCRETE:
            assert node.id_range is not None
            ids = range(node.id_range[0], node.id_range[1])
            print(f"std::string_view {node.name}KindName({node.name}Kind k) {{")
            print("  switch(k) {")
            for id in ids:
                name = node.Root().leaves[id].name
                desc = " ".join(
                    w.lower() for w in re.sub(r"([A-Z])", r" \1", name).split()
                )
                print(f'    case {node.name}Kind::{name}: return "{desc}";')
            print("  }")
            print("}\n")

    print("}  // namespace Carbon\n")
    cpp_file.close()


if __name__ == "__main__":
    main()
