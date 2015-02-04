{
  for (int c1 = 1; c1 < O; c1 += 1)
    for (int c3 = Q; c3 < N; c3 += 1)
      for (int c5 = P; c5 < M; c5 += 1)
        S1(c1, c3, c5);
  for (int c1 = 1; c1 < O; c1 += 1)
    for (int c3 = Q; c3 < N; c3 += 1)
      for (int c5 = 1; c5 < M; c5 += 1)
        S2(c1, c3, c5);
  for (int c1 = 1; c1 < O; c1 += 1)
    for (int c3 = 1; c3 < N; c3 += 1)
      for (int c5 = P; c5 < M; c5 += 1)
        S3(c1, c3, c5);
  for (int c1 = 1; c1 < O; c1 += 1)
    for (int c3 = 1; c3 < N; c3 += 1)
      for (int c5 = 1; c5 < M; c5 += 1)
        S4(c1, c3, c5);
}
