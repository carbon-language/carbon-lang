{
  if (N >= 2)
    for (int c1 = 1; c1 < O; c1 += 1) {
      for (int c5 = 1; c5 <= M; c5 += 1)
        S1(c1, 1, c5);
      for (int c5 = 1; c5 < M; c5 += 1) {
        S6(c1, 1, c5);
        S7(c1, 1, c5);
      }
      if (N >= 3) {
        for (int c5 = 1; c5 <= M; c5 += 1)
          S3(c1, 1, c5);
        for (int c5 = 1; c5 <= M; c5 += 1)
          S1(c1, 2, c5);
        for (int c5 = 1; c5 < M; c5 += 1) {
          S6(c1, 2, c5);
          S7(c1, 2, c5);
        }
        for (int c5 = 1; c5 < M; c5 += 1)
          S11(c1, 1, c5);
      } else {
        for (int c5 = 1; c5 <= M; c5 += 1)
          S3(c1, 1, c5);
        for (int c5 = 1; c5 < M; c5 += 1)
          S11(c1, 1, c5);
      }
      for (int c3 = 3; c3 < 2 * N - 4; c3 += 2) {
        for (int c5 = 1; c5 < M; c5 += 1)
          S10(c1, (c3 - 1) / 2, c5);
        for (int c5 = 1; c5 <= M; c5 += 1)
          S3(c1, (c3 + 1) / 2, c5);
        for (int c5 = 1; c5 <= M; c5 += 1)
          S1(c1, (c3 + 3) / 2, c5);
        for (int c5 = 1; c5 < M; c5 += 1) {
          S6(c1, (c3 + 3) / 2, c5);
          S7(c1, (c3 + 3) / 2, c5);
        }
        for (int c5 = 1; c5 < M; c5 += 1)
          S11(c1, (c3 + 1) / 2, c5);
      }
      if (N >= 3) {
        for (int c5 = 1; c5 < M; c5 += 1)
          S10(c1, N - 2, c5);
        for (int c5 = 1; c5 <= M; c5 += 1)
          S3(c1, N - 1, c5);
        for (int c5 = 1; c5 < M; c5 += 1)
          S11(c1, N - 1, c5);
      }
      for (int c5 = 1; c5 < M; c5 += 1)
        S10(c1, N - 1, c5);
    }
  for (int c1 = 1; c1 < O; c1 += 1)
    for (int c3 = 1; c3 < N; c3 += 1) {
      for (int c5 = 1; c5 <= M; c5 += 1)
        S2(c1, c3, c5);
      for (int c5 = 1; c5 < M; c5 += 1)
        S8(c1, c3, c5);
      for (int c5 = 1; c5 < M; c5 += 1)
        S9(c1, c3, c5);
    }
  for (int c1 = 1; c1 < O; c1 += 1)
    for (int c3 = 1; c3 < N; c3 += 1)
      for (int c5 = 1; c5 < M; c5 += 1)
        S4(c1, c3, c5);
  for (int c1 = 1; c1 < O; c1 += 1)
    for (int c3 = 1; c3 < N; c3 += 1)
      for (int c5 = 1; c5 < M; c5 += 1)
        S5(c1, c3, c5);
  for (int c1 = R; c1 < O; c1 += 1)
    for (int c3 = Q; c3 < N; c3 += 1)
      for (int c5 = P; c5 < M; c5 += 1)
        S12(c1, c3, c5);
  for (int c1 = R; c1 < O; c1 += 1)
    for (int c3 = Q; c3 < N; c3 += 1)
      for (int c5 = 1; c5 < M; c5 += 1)
        S13(c1, c3, c5);
  for (int c1 = R; c1 < O; c1 += 1)
    for (int c3 = 1; c3 < N; c3 += 1)
      for (int c5 = P; c5 < M; c5 += 1)
        S14(c1, c3, c5);
  for (int c1 = R; c1 < O; c1 += 1)
    for (int c3 = 1; c3 < N; c3 += 1)
      for (int c5 = 1; c5 < M; c5 += 1)
        S15(c1, c3, c5);
}
