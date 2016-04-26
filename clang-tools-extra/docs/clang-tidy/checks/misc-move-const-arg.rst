.. title:: clang-tidy - misc-move-const-arg

misc-move-const-arg
===================

The check warns if the result of ``std::move(x)`` is bound to a constant
reference argument, e.g.:

.. code:: c++

  void f(const string&);
  void g() {
    string s;
    F(std::move(s));  // Warning here. std::move() is not moving anything.
  }
