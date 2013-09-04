void update(int (&arr)[10]) {
  int val = 1;
  for (unsigned i = 0; i < sizeof(arr)/sizeof(int); ++i) {
    arr[i] = val++;
    // CHECK: for (auto & elem : arr) {
    // CHECK-NEXT: elem = val++;
  }
}
