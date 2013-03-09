.. index:: Loop Convert Transform

======================
Loop Convert Transform
======================

The Loop Convert Transform is a transformation to convert ``for(...; ...;
...)`` loops to use the new range-based loops in C++11. The transform is enabled
with the :option:`-loop-convert` option of :program:`cpp11-migrate`.

Three kinds of loops can be converted:

-  Loops over statically allocated arrays
-  Loops over containers, using iterators
-  Loops over array-like containers, using ``operator[]`` and ``at()``

Risk
====

TODO: Add code examples for which incorrect transformations are performed
when the risk level is set to "Risky" or "Reasonable".

Risky
^^^^^

In loops where the container expression is more complex than just a
reference to a declared expression (a variable, function, enum, etc.),
and some part of it appears elsewhere in the loop, we lower our confidence
in the transformation due to the increased risk of changing semantics.
Transformations for these loops are marked as `risky`, and thus will only
be converted if the acceptable risk level is set to ``-risk=risky``.

.. code-block:: c++

  int arr[10][20];
  int l = 5;

  for (int j = 0; j < 20; ++j)
    int k = arr[l][j] + l; // using l outside arr[l] is considered risky

  for (int i = 0; i < obj.getVector().size(); ++i)
    obj.foo(10); // using 'obj' is considered risky

Reasonable (Default)
^^^^^^^^^^^^^^^^^^^^

If a loop calls ``.end()`` or ``.size()`` after each iteration, the
transformation for that loop is marked as `reasonable`, and thus will
be converted if the acceptable risk level is set to ``-risk=reasonable``
(default) or higher.

.. code-block:: c++

  // using size() is considered reasonable
  for (int i = 0; i < container.size(); ++i)
    cout << container[i];

Safe
^^^^

Any other loops that do not match the above criteria to be marked as
`risky` or `reasonable` are marked `safe`, and thus will be converted
if the acceptable risk level is set to ``-risk=safe`` or higher.

.. code-block:: c++

  int arr[] = {1,2,3};

  for (int i = 0; i < 3; ++i)
    cout << arr[i];

Example
=======

Original:

.. code-block:: c++

  const int N = 5;
  int arr[] = {1,2,3,4,5};
  vector<int> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);

  // safe transform
  for (int i = 0; i < N; ++i)
    cout << arr[i];

  // reasonable transform
  for (vector<int>::iterator it = v.begin(); it != v.end(); ++it)
    cout << *it;*

  // reasonable transform
  for (int i = 0; i < v.size(); ++i)
    cout << v[i];

After transformation with risk level set to ``-risk=reasonable`` (default):

.. code-block:: c++

  const int N = 5;
  int arr[] = {1,2,3,4,5};
  vector<int> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);

  // safe transform
  for (auto & elem : arr)
    cout << elem;

  // reasonable transform
  for (auto & elem : v)
    cout << elem;

  // reasonable transform
  for (auto & elem : v)
    cout << elem;

