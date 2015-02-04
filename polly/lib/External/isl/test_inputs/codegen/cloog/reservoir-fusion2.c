if (N >= 1) {
  for (int c3 = 1; c3 <= M; c3 += 1)
    S1(1, c3);
  for (int c1 = 2; c1 <= N; c1 += 1) {
    for (int c3 = 1; c3 <= M; c3 += 1)
      S2(c1 - 1, c3);
    for (int c3 = 1; c3 <= M; c3 += 1)
      S1(c1, c3);
  }
  for (int c3 = 1; c3 <= M; c3 += 1)
    S2(N, c3);
}
