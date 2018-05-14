
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

void printCluster(const std::vector<InstructionBenchmark> &Points,
                  const llvm::MCSubtargetInfo &STI,
                  const size_t ClusterId,
                  const InstructionBenchmarkClustering::Cluster &Cluster,
                  llvm::raw_ostream &OS) {
  // TODO:
  // GetSchedClass(Points[PointIdB]); });

  // Print all points.
  for (const auto &PointId : Cluster.PointIndices) {
    renderInstructionRow(Points[PointId], NameLen, OS);
  }
}

} // namespace

llvm::Error
printAnalysisClusters(const InstructionBenchmarkClustering &Clustering,
                      const llvm::MCSubtargetInfo &STI, llvm::raw_ostream &OS) {
  OS << "cluster_id,key,";
  for (size_t I = 0, E = Clustering.getValidClusters().size(); I < E; ++I) {
    printCluster(Clustering.getPoints(), STI, I, Clustering.getValidClusters()[I], OS);
    OS << "\n\n";
  }

  return llvm::Error::success();
}

} // namespace exegesis
