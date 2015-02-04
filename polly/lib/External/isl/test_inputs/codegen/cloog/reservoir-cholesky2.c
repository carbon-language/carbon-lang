for (int c1 = 2; c1 < 3 * M; c1 += 1) {
  if ((c1 - 2) % 3 == 0)
    S1((c1 + 1) / 3);
  for (int c3 = (c1 + 1) / 3 + 1; c3 <= min(M, c1 - 2); c3 += 1)
    for (int c5 = -c3 + (c1 + c3 + 1) / 2 + 1; c5 <= min(c3, c1 - c3); c5 += 1)
      S3(c1 - c3 - c5 + 1, c3, c5);
  for (int c3 = -c1 + 2 * ((2 * c1 + 1) / 3) + 2; c3 <= min(M, c1); c3 += 2)
    S2(((c1 - c3) / 2) + 1, c3);
}
