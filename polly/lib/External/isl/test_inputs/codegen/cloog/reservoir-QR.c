if (N >= 1) {
  S1(0);
  if (N == 1) {
    for (int c3 = 0; c3 < M; c3 += 1)
      S2(0, c3);
    S3(0);
    for (int c3 = 0; c3 < M; c3 += 1)
      S4(0, c3);
    S10(0);
    S5(0);
  } else {
    for (int c3 = 0; c3 < M; c3 += 1)
      S2(0, c3);
    S3(0);
    for (int c3 = 0; c3 < M; c3 += 1)
      S4(0, c3);
    S10(0);
    S1(1);
    S5(0);
  }
  for (int c1 = 2; c1 < N; c1 += 1) {
    for (int c3 = c1 - 1; c3 < N; c3 += 1) {
      S6(c1 - 2, c3);
      for (int c5 = c1 - 2; c5 < M; c5 += 1)
        S7(c1 - 2, c3, c5);
      S8(c1 - 2, c3);
      for (int c5 = c1 - 2; c5 < M; c5 += 1)
        S9(c1 - 2, c3, c5);
    }
    for (int c3 = c1 - 1; c3 < M; c3 += 1)
      S2(c1 - 1, c3);
    S3(c1 - 1);
    for (int c3 = c1 - 1; c3 < M; c3 += 1)
      S4(c1 - 1, c3);
    S10(c1 - 1);
    S1(c1);
    S5(c1 - 1);
  }
  if (N >= 2) {
    S6(N - 2, N - 1);
    for (int c5 = N - 2; c5 < M; c5 += 1)
      S7(N - 2, N - 1, c5);
    S8(N - 2, N - 1);
    for (int c5 = N - 2; c5 < M; c5 += 1)
      S9(N - 2, N - 1, c5);
    for (int c3 = N - 1; c3 < M; c3 += 1)
      S2(N - 1, c3);
    S3(N - 1);
    for (int c3 = N - 1; c3 < M; c3 += 1)
      S4(N - 1, c3);
    S10(N - 1);
    S5(N - 1);
  }
}
