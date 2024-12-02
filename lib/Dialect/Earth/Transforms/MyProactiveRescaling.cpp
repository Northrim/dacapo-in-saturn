
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"

#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_MYPROACTIVERESCALING
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct MyProactiveRescalingPass
    : public hecate::earth::impl::MyProactiveRescalingBase<
          MyProactiveRescalingPass> {
  MyProactiveRescalingPass() {}

  MyProactiveRescalingPass(hecate::earth::MyProactiveRescalingOptions ops) {
    this->waterline = ops.waterline;
    this->output_val = ops.output_val;
    this->in_scale = ops.in_scale;
    this->in_level = ops.in_level;
  }

  void runOnOperation() override {

    auto func = getOperation();

    markAnalysesPreserved<hecate::ScaleManagementUnit>();

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);
    SmallVector<mlir::Type, 4> inputTypes;

    // Hard setting the input scale
    for (auto argval : func.getArguments()) {
      auto tt = argval.getType().dyn_cast<RankedTensorType>();
      argval.setType(RankedTensorType::get(
          tt.getShape(), tt.getElementType()
                             .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                             .switchScale(in_scale)
                             .switchLevel(in_level)));
      inputTypes.push_back(argval.getType());
    }

    // Skip This
    // hecate::earth::refineInputValues(func, builder, inputTypes, waterline,
    //                                  output_val);

    llvm::outs() << "William: We are in the MY PROACTIVE RESCALE.\n";
    llvm::outs() << "inputTypes: " << inputTypes << "\n";

    // Apply waterline rescaling for the operations
    func.walk([&](hecate::earth::ForwardMgmtInterface sop) {
      builder.setInsertionPointAfter(sop.getOperation());
      sop.processOperandsPARS(waterline);
      inferTypeForward(sop);
      sop.processResultsPARS(waterline);
    });
    hecate::earth::refineReturnValues(func, builder, inputTypes, waterline,
                                      output_val);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
