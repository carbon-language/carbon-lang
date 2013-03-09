.. index:: Use-Nullptr Transform

=====================
Use-Nullptr Transform
=====================

The Use-Nullptr Transform is a transformation to convert the usage of null
pointer constants (eg. ``NULL``, ``0``) to use the new C++11 ``nullptr``
keyword. The transform is enabled with the :option:`-use-nullptr` option of
:program:`cpp11-migrate`.

Example
=======

.. code-block:: c++

  void assignment() {
    char *a = NULL;
    char *b = 0;
    char c = 0;
  }

  int *ret_ptr() {
    return 0;
  }


transforms to:

.. code-block:: c++

  void assignment() {
    char *a = nullptr;
    char *b = nullptr;
    char c = 0;
  }

  int *ret_ptr() {
    return nullptr;
  }
