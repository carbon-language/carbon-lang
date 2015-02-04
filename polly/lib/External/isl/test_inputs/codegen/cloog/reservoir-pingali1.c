for (int c1 = 1; c1 <= M; c1 += 1)
  for (int c3 = 1; c3 < 2 * N; c3 += 1) {
    for (int c5 = max(1, -N + c3); c5 < (c3 + 1) / 2; c5 += 1)
      S1(c1, c3 - c5, c5);
    if ((c3 - 1) % 2 == 0)
      S2(c1, (c3 + 1) / 2);
  }
