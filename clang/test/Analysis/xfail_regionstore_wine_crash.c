// RUN: clang-cc -checker-cfref -analyze -analyzer-store=region -verify %s
// XFAIL

// When this test passes we should put it in the misc-ps.m test file.
// This test fails now because RegionStoreManager::Retrieve() does correctly 
// retrieve the first byte of 'x' when retrieving '*y'.
void foo() {
  long x = 0;
  char *y = (char *) &x;
  if (!*y)
    return;
}
