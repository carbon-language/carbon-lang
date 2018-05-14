
#include "Analysis.h"
#include "llvm/Support/Format.h"

namespace exegesis {

namespace {

// Prints a row representing an instruction, along with scheduling info and
// point coordinates (measurements).
void renderInstructionRow(const InstructionBenchmark &Point,
                          const size_t NameLen, llvm::raw_ostream &OS) {
  OS << llvm::format("%*s", NameLen, Point.AsmTmpl.Name.c_str());
  for (const auto &Measurement : Point.Measurements) {
    OS << llvm::format("   %*.2f", Measurement.Key.size(), Measurement.Value);
  }
  OS << "\n";
}

void analyzeCluster(const std::vector<InstructionBenchmark> &Points,
                    const llvm::MCSubtargetInfo &STI,
                    const InstructionBenchmarkClustering::Cluster &Cluster,
                    llvm::raw_ostream &OS) {
  // TODO:
  // std::sort(Cluster.PointIndices.begin(), Cluster.PointIndices.end(),
  // [](int PointIdA, int PointIdB) { return GetSchedClass(Points[PointIdA]) <
  // GetSchedClass(Points[PointIdB]); });
  OS << "Cluster:\n";
  // Get max length of the name for alignement.
  size_t NameLen = 0;
  for (const auto &PointId : Cluster.PointIndices) {
    NameLen = std::max(NameLen, Points[PointId].AsmTmpl.Name.size());
  }

  // Print all points.
  for (const auto &PointId : Cluster.PointIndices) {
    renderInstructionRow(Points[PointId], NameLen, OS);
  }
}

} // namespace

llvm::Error
printAnalysisClusters(const InstructionBenchmarkClustering &Clustering,
                      const llvm::MCSubtargetInfo &STI, llvm::raw_ostream &OS) {

  for (const auto &Cluster : Clustering.getValidClusters()) {
    analyzeCluster(Clustering.getPoints(), STI, Cluster, OS);
    OS << "\n\n\n";
  }

  return llvm::Error::success();
}

} // namespace exegesis
