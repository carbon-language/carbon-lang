
Meta TODO list:
1. Create a new DAG -> DAG instruction selector, by adding patterns to the
   instructions.
2. ???
3. profit!

To-do
-----

* open code 64-bit shifts
* Keep the address of the constant pool in a register instead of forming its
  address all of the time.
* We can fold small constant offsets into the %hi/%lo references to constant
  pool addresses as well.

