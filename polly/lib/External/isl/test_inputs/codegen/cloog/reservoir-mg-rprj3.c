if (M >= 2 && N >= 3)
  for (int c1 = 2; c1 < O; c1 += 1) {
    for (int c5 = 2; c5 <= M; c5 += 1)
      S1(c1, 2, c5);
    for (int c3 = 3; c3 < N; c3 += 1) {
      for (int c5 = 2; c5 <= M; c5 += 1)
        S2(c1, c3 - 1, c5);
      if (M >= 3) {
        S4(c1, c3 - 1, 2);
        for (int c5 = 2; c5 < M - 1; c5 += 1) {
          S3(c1, c3 - 1, c5);
          S5(c1, c3 - 1, c5);
          S4(c1, c3 - 1, c5 + 1);
        }
        S3(c1, c3 - 1, M - 1);
        S5(c1, c3 - 1, M - 1);
      }
      for (int c5 = 2; c5 <= M; c5 += 1)
        S1(c1, c3, c5);
    }
    for (int c5 = 2; c5 <= M; c5 += 1)
      S2(c1, N - 1, c5);
    if (M >= 3) {
      S4(c1, N - 1, 2);
      for (int c5 = 2; c5 < M - 1; c5 += 1) {
        S3(c1, N - 1, c5);
        S5(c1, N - 1, c5);
        S4(c1, N - 1, c5 + 1);
      }
      S3(c1, N - 1, M - 1);
      S5(c1, N - 1, M - 1);
    }
  }
