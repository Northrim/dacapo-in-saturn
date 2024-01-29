
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"

#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_PROACTIVERESCALING
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct ProactiveRescalingPass
    : public hecate::earth::impl::ProactiveRescalingBase<
          ProactiveRescalingPass> {
  ProactiveRescalingPass() {}

  ProactiveRescalingPass(hecate::earth::ProactiveRescalingOptions ops) {
    this->waterline = ops.waterline;
    this->output_val = ops.output_val;
  }

  void runOnOperation() override {

    auto func = getOperation();

    markAnalysesPreserved<hecate::ScaleManagementUnit>();

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);
    SmallVector<mlir::Type, 4> inputTypes;
    // Set function argument types
    if (!func->hasAttr("segment_inputType")) {
      for (auto argval : func.getArguments()) {
        argval.setType(
            argval.getType().dyn_cast<RankedTensorType>().replaceSubElements(
                [&](hecate::earth::HEScaleTypeInterface t) {
                  return t.switchScale(waterline);
                }));
        inputTypes.push_back(argval.getType());
      }
    } else {
      auto &&inputType_attrs = func->getAttr("segment_inputType")
                                   .dyn_cast<mlir::ArrayAttr>()
                                   .getValue();
      for (size_t i = 0; i < func.getNumArguments(); i++) {
        auto argval = func.getArgument(i);
        auto input_type = inputType_attrs[i]
                              .dyn_cast<mlir::TypeAttr>()
                              .getValue()
                              .dyn_cast<hecate::earth::HEScaleTypeInterface>();
        argval.setType(input_type);
        inputTypes.push_back(argval.getType());
      }
    }

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
