modernize-replace-auto-ptr
==========================


Transforms the deprecated ``std::auto_ptr`` into the C++11 ``std::unique_ptr``.

Note that both the ``std::auto_ptr`` type and the transfer of ownership are
transformed. ``std::auto_ptr`` provides two ways to transfer the ownership,
the copy-constructor and the assignment operator. Unlike most classes these
operations do not 'copy' the resource but they 'steal' it.
``std::unique_ptr`` uses move semantics instead, which makes the intent of
transferring the resource explicit. This difference between the two smart
pointers requeres to wrap the copy-ctor and assign-operator with
``std::move()``.

For example, given:

.. code:: c++

  std::auto_ptr<int> i, j;
  i = j;

This code is transformed to:

.. code:: c++

  std::unique_ptr<in> i, j;
  i = std::move(j);

