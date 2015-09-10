readability-inconsistent-declaration-parameter-name
===================================================


Find function declarations which differ in parameter names.

Example:

.. code:: c++

  // in foo.hpp:
  void foo(int a, int b, int c);

  // in foo.cpp:
  void foo(int d, int e, int f); // warning

This check should help to enforce consistency in large projects, where it often
happens that a definition of function is refactored, changing the parameter
names, but its declaration in header file is not updated. With this check, we
can easily find and correct such inconsistencies, keeping declaration and
definition always in sync.

Unnamed parameters are allowed and are not taken into account when comparing
function declarations, for example:

.. code:: c++

   void foo(int a);
   void foo(int); // no warning

If there are multiple declarations of same function, only one warning will be
generated.
