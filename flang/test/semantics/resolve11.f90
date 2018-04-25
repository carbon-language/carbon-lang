module m
  public i
  integer, private :: j
  !ERROR: The accessibility of 'i' has already been specified as PUBLIC
  private i
  !The accessibility of 'j' has already been specified as PRIVATE
  private j
end
