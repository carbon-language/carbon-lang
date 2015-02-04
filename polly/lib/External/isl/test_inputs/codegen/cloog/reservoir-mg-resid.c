for (int c1 = 2; c1 < O; c1 += 1)
  for (int c3 = 3; c3 < 2 * N - 2; c3 += 2) {
    for (int c5 = 1; c5 <= M; c5 += 1) {
      S1(c1, (c3 + 1) / 2, c5);
      S2(c1, (c3 + 1) / 2, c5);
    }
    for (int c5 = 2; c5 < M; c5 += 1)
      S3(c1, (c3 + 1) / 2, c5);
  }
