.. index:: Add-Override Transform

======================
Add-Override Transform
======================

The Add-Override Transform adds the ``override`` specifier to member
functions that override a virtual function in a base class and that
don't already have the specifier. The transform is enabled with the 
:option:`-add-override` option of :program:`cpp11-migrate`.
For example:

.. code-block:: c++

  class A {
  public:
    virtual void h() const;
  };

  class B : public A {
  public:
    void h() const;

    // The declaration of h is transformed to
    void h() const override;
  };

Using Expands-to-Override Macros
================================

Like LLVM's ``LLVM_OVERRIDE``, several projects have macros that conditionally
expand to the ``override`` keyword when compiling with C++11 features enabled.
To maintain compatibility with non-C++11 builds, the Add-Override Transform
supports detection and use of these macros instead of using the ``override``
keyword directly. Specify ``-override-macros`` on the command line to the
Migrator to enable this behavior.


Known Limitations
=================
* This transform will not insert the override keyword if a method is
  pure. At the moment it's not possible to track down the pure
  specifier location.

.. code-block:: c++

  class B : public A {
  public:
    virtual void h() const = 0;

    // The declaration of h is NOT transformed to
    virtual void h() const override = 0;
  };

