.. title:: clang-tidy - llvm-twine-local

llvm-twine-local
================


Looks for local ``Twine`` variables which are prone to use after frees and
should be generally avoided.

.. code-block:: c++

  static Twine Moo = Twine("bark") + "bah";

  // becomes

  static std::string Moo = (Twine("bark") + "bah").str();
