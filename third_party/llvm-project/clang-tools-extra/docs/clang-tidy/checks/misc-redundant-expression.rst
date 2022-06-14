.. title:: clang-tidy - misc-redundant-expression

misc-redundant-expression
=========================

Detect redundant expressions which are typically errors due to copy-paste.

Depending on the operator expressions may be

- redundant,

- always ``true``,

- always ``false``,

- always a constant (zero or one).

Examples:

.. code-block:: c++

  ((x+1) | (x+1))             // (x+1) is redundant
  (p->x == p->x)              // always true
  (p->x < p->x)               // always false
  (speed - speed + 1 == 12)   // speed - speed is always zero
