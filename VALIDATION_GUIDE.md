# ALS CP Decomposition Validation Guide

## How to Verify Your ALS Implementation Works Correctly

This guide explains how to validate that your `als_cp_decomposition.jl` implementation is optimizing properly.

## Key Indicators of Proper Optimization

### 1. **Recovery of Known Factors** ✓
- **Test**: Create a tensor from known factors, then decompose it
- **Expected**: Reconstruction error < 1e-6 for noise-free data
- **What it shows**: The algorithm can find the correct solution when it exists

### 2. **Monotonic Error Decrease** ✓
- **Test**: Track error at each iteration
- **Expected**: Error should decrease (or stay constant) at each iteration
- **What it shows**: The optimization is making progress, not diverging

### 3. **Convergence Behavior** ✓
- **Test**: Run with different tolerance levels
- **Expected**: 
  - Converges within max_iter for reasonable tolerances
  - Stricter tolerances require more iterations
- **What it shows**: The algorithm can reach the desired precision

### 4. **Noisy Data Recovery** ✓
- **Test**: Add noise to known tensor and decompose
- **Expected**: Error scales approximately with noise level
- **What it shows**: The algorithm is robust to noise

### 5. **Different Tensor Shapes** ✓
- **Test**: Test on various tensor dimensions and ranks
- **Expected**: Works for 3-way, 4-way, etc., and different ranks
- **What it shows**: General applicability

### 6. **Rank Mismatch Handling** ✓
- **Test**: Decompose with wrong rank
- **Expected**: 
  - Under-estimated rank → higher error
  - Over-estimated rank → low error (but may overfit)
- **What it shows**: Sensible behavior when rank is unknown

## Running the Validation Tests

```bash
julia test_als_cp.jl
```

## Test Results Interpretation

### ✅ All Tests Pass
If all tests show "✓ PASS", your implementation is working correctly:
- Optimization is converging properly
- Error decreases monotonically
- Can recover known factors
- Handles noise appropriately

### ⚠️ Warnings
- **High error on noise-free data**: Check numerical stability
- **Non-monotonic error**: May indicate bugs in update equations
- **No convergence**: May need more iterations or different initialization

### ❌ Failures
- **Cannot recover known factors**: Fundamental algorithm issue
- **Error increases**: Bug in optimization step
- **Doesn't work on simple cases**: Implementation error

## What Each Test Validates

### Test 1: Factor Recovery
```julia
# Create tensor from known factors
tensor = cp_reconstruct(factors_true, weights_true)
# Decompose
result = als_cp_decomposition(tensor, rank)
# Check: result[:relative_error] < 1e-6
```
**Validates**: Core algorithm correctness

### Test 2: Monotonic Decrease
```julia
# Track error at each iteration
error_history = [error_1, error_2, ..., error_n]
# Check: error[i] <= error[i-1] (with small tolerance)
```
**Validates**: Optimization progress

### Test 3: Convergence
```julia
# Test different tolerances
for tol in [1e-4, 1e-6, 1e-8]
    result = als_cp_decomposition(tensor, rank; tol=tol)
    # Check: result[:converged] == true
end
```
**Validates**: Convergence criterion works

### Test 4: Noise Robustness
```julia
# Add noise
tensor_noisy = tensor + noise
result = als_cp_decomposition(tensor_noisy, rank)
# Check: error ≈ noise_level
```
**Validates**: Robustness to real-world data

## Common Issues and Solutions

### Issue: High Reconstruction Error
**Possible causes**:
- Rank too low for the data
- Need more iterations
- Numerical precision issues

**Solutions**:
- Increase rank
- Increase max_iter
- Check tensor normalization

### Issue: Non-Monotonic Error
**Possible causes**:
- Bug in factor update
- Numerical instability
- Normalization issues

**Solutions**:
- Check Khatri-Rao product computation
- Verify pseudo-inverse calculation
- Add regularization

### Issue: No Convergence
**Possible causes**:
- Tolerance too strict
- Local minimum
- Rank mismatch

**Solutions**:
- Relax tolerance
- Try different initialization
- Check if rank is appropriate

## Real-World Validation

For your `moments.jl` application:

1. **Check relative errors**: Should be reasonable (2-10% for rank-5 on 4th-order tensors)
2. **Monitor iterations**: Should converge within 1000 iterations
3. **Compare groups**: Similar groups should have similar factor patterns
4. **Component weights**: Should be interpretable (not all zeros or identical)

## Performance Benchmarks

Expected performance on a 28×28×28×28 tensor with rank 5:
- **Iterations**: 200-1000 (depending on tolerance)
- **Time**: Seconds to minutes (depends on hardware)
- **Memory**: Moderate (stores full tensor and factors)

## Conclusion

If all validation tests pass, you can be confident that:
1. ✅ The ALS algorithm is implemented correctly
2. ✅ Optimization is working (error decreases)
3. ✅ The implementation is robust and general
4. ✅ It will work on your real data

The test suite provides comprehensive validation of your implementation!

