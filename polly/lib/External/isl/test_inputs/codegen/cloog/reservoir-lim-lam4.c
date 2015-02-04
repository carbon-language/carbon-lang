for (int c1 = 1; c1 < 2 * M - 1; c1 += 1) {
  for (int c3 = max(-M + 1, -c1 + 1); c3 < 0; c3 += 1) {
    for (int c7 = max(1, -M + c1 + 1); c7 <= min(M - 1, c1 + c3); c7 += 1)
      S1(c7, c1 + c3 - c7, -c3);
    for (int c5 = max(-M + c1 + 1, -c3); c5 < min(M, c1); c5 += 1)
      S2(c1 - c5, c3 + c5, c5);
  }
  for (int c7 = max(1, -M + c1 + 1); c7 <= min(M - 1, c1); c7 += 1)
    S1(c7, c1 - c7, 0);
}
