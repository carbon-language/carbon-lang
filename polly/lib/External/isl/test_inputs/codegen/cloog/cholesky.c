for (int c0 = 1; c0 <= n; c0 += 1) {
  S1(c0);
  for (int c2 = 1; c2 < c0; c2 += 1)
    S2(c0, c2);
  S3(c0);
  for (int c2 = c0 + 1; c2 <= n; c2 += 1) {
    S4(c0, c2);
    for (int c4 = 1; c4 < c0; c4 += 1)
      S5(c0, c2, c4);
    S6(c0, c2);
  }
}
