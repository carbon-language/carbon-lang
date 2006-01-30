
To-do
-----

* Keep the address of the constant pool in a register instead of forming its
  address all of the time.
* We can fold small constant offsets into the %hi/%lo references to constant
  pool addresses as well.
* When in V9 mode, register allocate %icc[0-3].

