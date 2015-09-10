misc-sizeof-container
=====================

The check finds usages of ``sizeof`` on expressions of STL container types. Most
likely the user wanted to use ``.size()`` instead.

Currently only ``std::string`` and ``std::vector<T>`` are supported.

Examples:

.. code:: c++

  std::string s;
  int a = 47 + sizeof(s); // warning: sizeof() doesn't return the size of the container. Did you mean .size()?
                          // The suggested fix is: int a = 47 + s.size();

  int b = sizeof(std::string); // no warning, probably intended.

  std::string array_of_strings[10];
  int c = sizeof(array_of_strings) / sizeof(array_of_strings[0]); // no warning, definitely intended.
