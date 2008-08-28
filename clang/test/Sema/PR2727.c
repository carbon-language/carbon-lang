int f (int x)
{
  // sizeof applied to a type should not delete the type.
  return sizeof (int[x]);
}
