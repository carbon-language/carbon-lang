for (int c1 = 1; c1 <= M; c1 += 1)
  for (int c3 = 2; c3 <= M + c1; c3 += 1)
    for (int c5 = max(1, -c1 + c3); c5 <= min(M, c3 - 1); c5 += 1)
      S1(c1, c5, c3 - c5);
