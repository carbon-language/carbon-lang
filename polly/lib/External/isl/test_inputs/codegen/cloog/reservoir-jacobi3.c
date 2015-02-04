for (int c1 = 1; c1 <= M; c1 += 1) {
  for (int c3 = 2; c3 < N; c3 += 1)
    for (int c5 = 2; c5 < N; c5 += 1)
      S1(c1, c3, c5);
  for (int c3 = 2; c3 < N; c3 += 1)
    for (int c5 = 2; c5 < N; c5 += 1)
      S2(c1, c3, c5);
}
