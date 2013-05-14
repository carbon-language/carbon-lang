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


Known Limitations
-----------------
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

