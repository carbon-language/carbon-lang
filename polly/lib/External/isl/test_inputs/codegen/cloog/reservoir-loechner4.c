for (int c1 = 2; c1 <= 2 * M; c1 += 1)
  for (int c3 = 1; c3 <= M; c3 += 1)
    for (int c5 = 1; c5 <= M; c5 += 1)
      for (int c7 = max(1, -M + c1); c7 <= min(M, c1 - 1); c7 += 1)
        S1(c5, c3, c7, c1 - c7);
