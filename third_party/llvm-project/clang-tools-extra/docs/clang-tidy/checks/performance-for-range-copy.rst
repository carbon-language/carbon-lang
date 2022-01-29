.. title:: clang-tidy - performance-for-range-copy

performance-for-range-copy
==========================

Finds C++11 for ranges where the loop variable is copied in each iteration but
it would suffice to obtain it by const reference.

The check is only applied to loop variables of types that are expensive to copy
which means they are not trivially copyable or have a non-trivial copy
constructor or destructor.

To ensure that it is safe to replace the copy with a const reference the
following heuristic is employed:

1. The loop variable is const qualified.
2. The loop variable is not const, but only const methods or operators are
   invoked on it, or it is used as const reference or value argument in
   constructors or function calls.

Options
-------

.. option:: WarnOnAllAutoCopies

   When `true`, warns on any use of `auto` as the type of the range-based for
   loop variable. Default is `false`.

.. option:: AllowedTypes

   A semicolon-separated list of names of types allowed to be copied in each
   iteration. Regular expressions are accepted, e.g. `[Rr]ef(erence)?$` matches
   every type with suffix `Ref`, `ref`, `Reference` and `reference`. The default
   is empty. If a name in the list contains the sequence `::` it is matched
   against the qualified typename (i.e. `namespace::Type`, otherwise it is
   matched against only the type name (i.e. `Type`).
