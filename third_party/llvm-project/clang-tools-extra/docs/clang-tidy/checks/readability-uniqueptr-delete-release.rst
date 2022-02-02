.. title:: clang-tidy - readability-uniqueptr-delete-release

readability-uniqueptr-delete-release
====================================

Replace ``delete <unique_ptr>.release()`` with ``<unique_ptr> = nullptr``.
The latter is shorter, simpler and does not require use of raw pointer APIs.

.. code-block:: c++

  std::unique_ptr<int> P;
  delete P.release();

  // becomes

  std::unique_ptr<int> P;
  P = nullptr;

Options
-------

.. option:: PreferResetCall

  If `true`, refactor by calling the reset member function instead of
  assigning to ``nullptr``. Default value is `false`.

  .. code-block:: c++

   std::unique_ptr<int> P;
   delete P.release();

   // becomes

   std::unique_ptr<int> P;
   P.reset();
