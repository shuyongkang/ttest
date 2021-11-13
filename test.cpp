// File: gen.cc
#include <random>
#include <iomanip>
#include <array>
#include <exception>

#include <tvm/tvm.h>
#include <tvm/operation.h>
#include <tvm/tensor.h>
#include <tvm/build_module.h>
#include <topi/broadcast.h>

using namespace std;

int main()
{
  /* Stage 1 */
  auto n = tvm::var("n");
  tvm::Array<tvm::Expr> shape = {n};
  tvm::Tensor A = tvm::placeholder(shape, tvm::Float(32), "A");
  tvm::Tensor B = tvm::placeholder(shape, tvm::Float(32), "B");
  tvm::Tensor C = tvm::compute(shape, tvm::FCompute([=](auto i){ return A(i) + B(i); } )) ;

  /* Stage 2 */
  tvm::Schedule s = tvm::create_schedule({C->op});

  /* Stage 3 */
  tvm::BuildConfig config = tvm::build_config();
  std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
  auto args = tvm::Array<tvm::Tensor>({A, B, C});
  auto lowered = tvm::lower(s, args, "vecadd", binds, config);
  cerr << lowered << endl;

  /* Stage 4 */
  auto target = tvm::Target::create("llvm");
  auto target_host = tvm::Target::create("llvm");
  tvm::runtime::Module mod = tvm::build(lowered, target, target_host, config);
  cout << mod->GetSource() << endl;

  return 0;
}

