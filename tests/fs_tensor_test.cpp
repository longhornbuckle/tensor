#include <gtest/gtest.h>
#include <experimental/linear_algebra.hpp>

namespace
{

/*
  TEST( DR_TENSOR, NEGATION )
  {
    using tensor_type = std::experimental::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Negate the tensor
    tensor_type negate_tensor { -tensor };
    // Access elements from const tensor
    auto val1 = std::experimental::math::detail::access( negate_tensor, 0, 0, 0 );
    auto val2 = std::experimental::math::detail::access( negate_tensor, 0, 0, 1 );
    auto val3 = std::experimental::math::detail::access( negate_tensor, 0, 1, 0 );
    auto val4 = std::experimental::math::detail::access( negate_tensor, 0, 1, 1 );
    auto val5 = std::experimental::math::detail::access( negate_tensor, 1, 0, 0 );
    auto val6 = std::experimental::math::detail::access( negate_tensor, 1, 0, 1 );
    auto val7 = std::experimental::math::detail::access( negate_tensor, 1, 1, 0 );
    auto val8 = std::experimental::math::detail::access( negate_tensor, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, -1.0 );
    EXPECT_EQ( val2, -2.0 );
    EXPECT_EQ( val3, -3.0 );
    EXPECT_EQ( val4, -4.0 );
    EXPECT_EQ( val5, -5.0 );
    EXPECT_EQ( val6, -6.0 );
    EXPECT_EQ( val7, -7.0 );
    EXPECT_EQ( val8, -8.0 );
  }

  TEST( DR_TENSOR, ADD )
  {
    using tensor_type = std::experimental::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Add the two tensors together
    tensor_type tensor_sum { tensor + tensor_copy };
    // Access elements from const tensor
    auto val1 = std::experimental::math::detail::access( tensor_sum, 0, 0, 0 );
    auto val2 = std::experimental::math::detail::access( tensor_sum, 0, 0, 1 );
    auto val3 = std::experimental::math::detail::access( tensor_sum, 0, 1, 0 );
    auto val4 = std::experimental::math::detail::access( tensor_sum, 0, 1, 1 );
    auto val5 = std::experimental::math::detail::access( tensor_sum, 1, 0, 0 );
    auto val6 = std::experimental::math::detail::access( tensor_sum, 1, 0, 1 );
    auto val7 = std::experimental::math::detail::access( tensor_sum, 1, 1, 0 );
    auto val8 = std::experimental::math::detail::access( tensor_sum, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( DR_TENSOR, ADD_ASSIGN )
  {
    using tensor_type = std::experimental::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Add the two tensors together
    static_cast<void>( tensor += tensor_copy );
    // Access elements from const tensor
    auto val1 = std::experimental::math::detail::access( tensor, 0, 0, 0 );
    auto val2 = std::experimental::math::detail::access( tensor, 0, 0, 1 );
    auto val3 = std::experimental::math::detail::access( tensor, 0, 1, 0 );
    auto val4 = std::experimental::math::detail::access( tensor, 0, 1, 1 );
    auto val5 = std::experimental::math::detail::access( tensor, 1, 0, 0 );
    auto val6 = std::experimental::math::detail::access( tensor, 1, 0, 1 );
    auto val7 = std::experimental::math::detail::access( tensor, 1, 1, 0 );
    auto val8 = std::experimental::math::detail::access( tensor, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( DR_TENSOR, SUBTRACT )
  {
    using tensor_type = std::experimental::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Subtract the two tensors
    tensor_type tensor_diff { tensor - tensor_copy };
    // Access elements from const tensor
    auto val1 = std::experimental::math::detail::access( tensor_diff, 0, 0, 0 );
    auto val2 = std::experimental::math::detail::access( tensor_diff, 0, 0, 1 );
    auto val3 = std::experimental::math::detail::access( tensor_diff, 0, 1, 0 );
    auto val4 = std::experimental::math::detail::access( tensor_diff, 0, 1, 1 );
    auto val5 = std::experimental::math::detail::access( tensor_diff, 1, 0, 0 );
    auto val6 = std::experimental::math::detail::access( tensor_diff, 1, 0, 1 );
    auto val7 = std::experimental::math::detail::access( tensor_diff, 1, 1, 0 );
    auto val8 = std::experimental::math::detail::access( tensor_diff, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0 );
    EXPECT_EQ( val2, 0 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 0 );
    EXPECT_EQ( val5, 0 );
    EXPECT_EQ( val6, 0 );
    EXPECT_EQ( val7, 0 );
    EXPECT_EQ( val8, 0 );
  }

  TEST( DR_TENSOR, SUBTRACT_ASSIGN )
  {
    using tensor_type = std::experimental::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Subtract the two tensors
    static_cast<void>( tensor -= tensor_copy );
    // Access elements from const tensor
    auto val1 = std::experimental::math::detail::access( tensor, 0, 0, 0 );
    auto val2 = std::experimental::math::detail::access( tensor, 0, 0, 1 );
    auto val3 = std::experimental::math::detail::access( tensor, 0, 1, 0 );
    auto val4 = std::experimental::math::detail::access( tensor, 0, 1, 1 );
    auto val5 = std::experimental::math::detail::access( tensor, 1, 0, 0 );
    auto val6 = std::experimental::math::detail::access( tensor, 1, 0, 1 );
    auto val7 = std::experimental::math::detail::access( tensor, 1, 1, 0 );
    auto val8 = std::experimental::math::detail::access( tensor, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0 );
    EXPECT_EQ( val2, 0 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 0 );
    EXPECT_EQ( val5, 0 );
    EXPECT_EQ( val6, 0 );
    EXPECT_EQ( val7, 0 );
    EXPECT_EQ( val8, 0 );
  }

  TEST( DR_TENSOR, SCALAR_PREMULTIPLY )
  {
    using tensor_type = std::experimental::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Pre multiply
    tensor_type tensor_prod { 2 * tensor };
    // Access elements from const tensor
    auto val1 = std::experimental::math::detail::access( tensor_prod, 0, 0, 0 );
    auto val2 = std::experimental::math::detail::access( tensor_prod, 0, 0, 1 );
    auto val3 = std::experimental::math::detail::access( tensor_prod, 0, 1, 0 );
    auto val4 = std::experimental::math::detail::access( tensor_prod, 0, 1, 1 );
    auto val5 = std::experimental::math::detail::access( tensor_prod, 1, 0, 0 );
    auto val6 = std::experimental::math::detail::access( tensor_prod, 1, 0, 1 );
    auto val7 = std::experimental::math::detail::access( tensor_prod, 1, 1, 0 );
    auto val8 = std::experimental::math::detail::access( tensor_prod, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( DR_TENSOR, SCALAR_POSTMULTIPLY )
  {
    using tensor_type = std::experimental::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Post multiply
    tensor_type tensor_prod { tensor * 2 };
    // Access elements from const tensor
    auto val1 = std::experimental::math::detail::access( tensor_prod, 0, 0, 0 );
    auto val2 = std::experimental::math::detail::access( tensor_prod, 0, 0, 1 );
    auto val3 = std::experimental::math::detail::access( tensor_prod, 0, 1, 0 );
    auto val4 = std::experimental::math::detail::access( tensor_prod, 0, 1, 1 );
    auto val5 = std::experimental::math::detail::access( tensor_prod, 1, 0, 0 );
    auto val6 = std::experimental::math::detail::access( tensor_prod, 1, 0, 1 );
    auto val7 = std::experimental::math::detail::access( tensor_prod, 1, 1, 0 );
    auto val8 = std::experimental::math::detail::access( tensor_prod, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( DR_TENSOR, SCALAR_MULTIPLY_ASSIGN )
  {
    using tensor_type = std::experimental::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Post multiply
    static_cast<void>( tensor *= 2 );
    // Access elements from tensor
    auto val1 = std::experimental::math::detail::access( tensor, 0, 0, 0 );
    auto val2 = std::experimental::math::detail::access( tensor, 0, 0, 1 );
    auto val3 = std::experimental::math::detail::access( tensor, 0, 1, 0 );
    auto val4 = std::experimental::math::detail::access( tensor, 0, 1, 1 );
    auto val5 = std::experimental::math::detail::access( tensor, 1, 0, 0 );
    auto val6 = std::experimental::math::detail::access( tensor, 1, 0, 1 );
    auto val7 = std::experimental::math::detail::access( tensor, 1, 1, 0 );
    auto val8 = std::experimental::math::detail::access( tensor, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( DR_TENSOR, SCALAR_DIVIDE )
  {
    using tensor_type = std::experimental::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Divide
    tensor_type tensor_divide { tensor / 2 };
    // Access elements from const tensor
    auto val1 = std::experimental::math::detail::access( tensor_divide, 0, 0, 0 );
    auto val2 = std::experimental::math::detail::access( tensor_divide, 0, 0, 1 );
    auto val3 = std::experimental::math::detail::access( tensor_divide, 0, 1, 0 );
    auto val4 = std::experimental::math::detail::access( tensor_divide, 0, 1, 1 );
    auto val5 = std::experimental::math::detail::access( tensor_divide, 1, 0, 0 );
    auto val6 = std::experimental::math::detail::access( tensor_divide, 1, 0, 1 );
    auto val7 = std::experimental::math::detail::access( tensor_divide, 1, 1, 0 );
    auto val8 = std::experimental::math::detail::access( tensor_divide, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0.5 );
    EXPECT_EQ( val2, 1.0 );
    EXPECT_EQ( val3, 1.5 );
    EXPECT_EQ( val4, 2.0 );
    EXPECT_EQ( val5, 2.5 );
    EXPECT_EQ( val6, 3.0 );
    EXPECT_EQ( val7, 3.5 );
    EXPECT_EQ( val8, 4.0 );
  }

  TEST( DR_TENSOR, SCALAR_DIVIDE_ASSIGN )
  {
    using tensor_type = std::experimental::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Divide
    static_cast<void>( tensor /= 2 );
    // Access elements from tensor
    auto val1 = std::experimental::math::detail::access( tensor, 0, 0, 0 );
    auto val2 = std::experimental::math::detail::access( tensor, 0, 0, 1 );
    auto val3 = std::experimental::math::detail::access( tensor, 0, 1, 0 );
    auto val4 = std::experimental::math::detail::access( tensor, 0, 1, 1 );
    auto val5 = std::experimental::math::detail::access( tensor, 1, 0, 0 );
    auto val6 = std::experimental::math::detail::access( tensor, 1, 0, 1 );
    auto val7 = std::experimental::math::detail::access( tensor, 1, 1, 0 );
    auto val8 = std::experimental::math::detail::access( tensor, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0.5 );
    EXPECT_EQ( val2, 1.0 );
    EXPECT_EQ( val3, 1.5 );
    EXPECT_EQ( val4, 2.0 );
    EXPECT_EQ( val5, 2.5 );
    EXPECT_EQ( val6, 3.0 );
    EXPECT_EQ( val7, 3.5 );
    EXPECT_EQ( val8, 4.0 );
  }
*/

  TEST( FS_TENSOR, DEFAULT_CONSTRUCTOR_AND_DESTRUCTOR )
  {
    // Default construction
    [[maybe_unused]] LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > > fs_tensor;
    // Destructor will be called when unit test ends and the fs tensor exits scope
  }

  TEST( FS_TENSOR, MUTABLE_AND_CONST_INDEX_ACCESS )
  {
    using fs_tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Default construct
    fs_tensor_type fs_tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( fs_tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( fs_tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( fs_tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( fs_tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( fs_tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( fs_tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( fs_tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Get a const reference
    const fs_tensor_type& const_fs_tensor( fs_tensor );
    // Access elements from const fs tensor
    auto val1 = LINALG_DETAIL::access( const_fs_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( const_fs_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( const_fs_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( const_fs_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( const_fs_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( const_fs_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( const_fs_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( const_fs_tensor, 1, 1, 1 );
    // Check the fs tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( FS_TENSOR, COPY_CONSTRUCTOR )
  {
    using fs_tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Default construct
    fs_tensor_type fs_tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( fs_tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( fs_tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( fs_tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( fs_tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( fs_tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( fs_tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( fs_tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    fs_tensor_type fs_tensor_copy{ fs_tensor };
    // Get a const reference to copy
    const fs_tensor_type& const_fs_tensor( fs_tensor_copy );
    // Access elements from const fs tensor
    auto val1 = LINALG_DETAIL::access( const_fs_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( const_fs_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( const_fs_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( const_fs_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( const_fs_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( const_fs_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( const_fs_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( const_fs_tensor, 1, 1, 1 );
    // Check the fs tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( FS_TENSOR, MOVE_CONSTRUCTOR )
  {
    using fs_tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Default construct
    fs_tensor_type fs_tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( fs_tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( fs_tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( fs_tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( fs_tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( fs_tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( fs_tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( fs_tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Move construct
    fs_tensor_type fs_tensor_move{ ::std::move( fs_tensor ) };
    // Get a const reference to moved tensor
    const fs_tensor_type& const_fs_tensor( fs_tensor_move );
    // Access elements from const fs tensor
    auto val1 = LINALG_DETAIL::access( const_fs_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( const_fs_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( const_fs_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( const_fs_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( const_fs_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( const_fs_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( const_fs_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( const_fs_tensor, 1, 1, 1 );
    // Check the fs tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( FS_TENSOR, CONSTRUCT_FROM_VIEW )
  {
    using fs_tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Default construct
    fs_tensor_type fs_tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( fs_tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( fs_tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( fs_tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( fs_tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( fs_tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( fs_tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( fs_tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Construct from view
    LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > > const_fs_tensor( ::std::mdspan( fs_tensor.data_handle(), fs_tensor.mapping(), fs_tensor.accessor() ) );
    // Access elements from const fs tensor
    auto val1 = LINALG_DETAIL::access( const_fs_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( const_fs_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( const_fs_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( const_fs_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( const_fs_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( const_fs_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( const_fs_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( const_fs_tensor, 1, 1, 1 );
    // Check the fs tensor tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }
  
  TEST( FS_TENSOR, TEMPLATE_COPY_CONSTRUCTOR )
  {
    using float_left_tensor_type   = LINALG::fs_tensor< float, ::std::extents< ::std::size_t, 2, 2, 2 >, ::std::experimental::layout_right, ::std::experimental::default_accessor<float> >;
    using double_right_tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 >, ::std::experimental::layout_left, ::std::experimental::default_accessor<double> >;
    // Default construct
    float_left_tensor_type fs_tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( fs_tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( fs_tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( fs_tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( fs_tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( fs_tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( fs_tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( fs_tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Construct from float tensor
    double_right_tensor_type fs_tensor_copy{ fs_tensor };
    // Access elements from const fs tensor tensor
    auto val1 = LINALG_DETAIL::access( fs_tensor_copy, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( fs_tensor_copy, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( fs_tensor_copy, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( fs_tensor_copy, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( fs_tensor_copy, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( fs_tensor_copy, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( fs_tensor_copy, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( fs_tensor_copy, 1, 1, 1 );
    // Check the fs tensor tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( FS_TENSOR, ASSIGNMENT_OPERATOR )
  {
    using fs_tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Default construct
    fs_tensor_type fs_tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( fs_tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( fs_tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( fs_tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( fs_tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( fs_tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( fs_tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( fs_tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Construct from lambda
    fs_tensor_type fs_tensor_copy;
    fs_tensor_copy = fs_tensor;
    // Access elements from const fs tensor
    auto val1 = LINALG_DETAIL::access( fs_tensor_copy, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( fs_tensor_copy, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( fs_tensor_copy, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( fs_tensor_copy, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( fs_tensor_copy, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( fs_tensor_copy, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( fs_tensor_copy, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( fs_tensor_copy, 1, 1, 1 );
    // Check the fs tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( FS_TENSOR, TEMPLATE_ASSIGNMENT_OPERATOR )
  {
    using float_left_tensor_type   = LINALG::fs_tensor< float, ::std::extents< ::std::size_t, 2, 2, 2 >, ::std::experimental::layout_right, ::std::experimental::default_accessor<float> >;
    using double_right_tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 >, ::std::experimental::layout_left, ::std::experimental::default_accessor<double> >;
    // Default construct
    float_left_tensor_type fs_tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( fs_tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( fs_tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( fs_tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( fs_tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( fs_tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( fs_tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( fs_tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Default construct and then assign
    double_right_tensor_type fs_tensor_copy;
    fs_tensor_copy = fs_tensor;
    // Access elements from const fs tensor
    auto val1 = LINALG_DETAIL::access( fs_tensor_copy, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( fs_tensor_copy, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( fs_tensor_copy, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( fs_tensor_copy, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( fs_tensor_copy, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( fs_tensor_copy, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( fs_tensor_copy, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( fs_tensor_copy, 1, 1, 1 );
    // Check the fs tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( FS_TENSOR, ASSIGN_FROM_VIEW )
  {
    using fs_tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Default construct
    fs_tensor_type fs_tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( fs_tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( fs_tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( fs_tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( fs_tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( fs_tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( fs_tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( fs_tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Default construct and assign from view
    auto fs_tensor_view = ::std::mdspan( fs_tensor.data_handle(), fs_tensor.mapping(), fs_tensor.accessor() );
    fs_tensor_type tensor_copy;
    tensor_copy = fs_tensor_view;
    // Get a const reference to constructed tensor
    const fs_tensor_type const_fs_tensor( tensor_copy );
    // Access elements from const fs tensor tensor
    auto val1 = LINALG_DETAIL::access( const_fs_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( const_fs_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( const_fs_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( const_fs_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( const_fs_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( const_fs_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( const_fs_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( const_fs_tensor, 1, 1, 1 );
    // Check the fs tensor tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( FS_TENSOR, SIZE )
  {
    using fs_tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 5, 1, 7 > >;
    // Default construct
    fs_tensor_type fs_tensor;
    EXPECT_TRUE( ( fs_tensor.extents().extent(0) == 2 ) );
    EXPECT_TRUE( ( fs_tensor.extents().extent(1) == 5 ) );
    EXPECT_TRUE( ( fs_tensor.extents().extent(2) == 1 ) );
    EXPECT_TRUE( ( fs_tensor.extents().extent(3) == 7 ) );
  }

  TEST( FS_TENSOR, CONST_SUBVECTOR )
  {
    // Construct
    LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 5, 5, 5 >, ::std::experimental::layout_right, ::std::experimental::default_accessor<double> > fs_tensor{ };
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          LINALG_DETAIL::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const auto& const_fs_tensor( fs_tensor );
    auto subvector = LINALG::subvector( const_fs_tensor, 0, ::std::full_extent, 1 );
    
    EXPECT_EQ( ( LINALG_DETAIL::access( subvector, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 0, 0, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subvector, 1 ) ), ( LINALG_DETAIL::access( fs_tensor, 0, 1, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subvector, 2 ) ), ( LINALG_DETAIL::access( fs_tensor, 0, 2, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subvector, 3 ) ), ( LINALG_DETAIL::access( fs_tensor, 0, 3, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subvector, 4 ) ), ( LINALG_DETAIL::access( fs_tensor, 0, 4, 1 ) ) );
  }

  TEST( FS_TENSOR, CONST_SUBMATRIX )
  {
    // Construct
    LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 5, 5, 5 >, ::std::experimental::layout_right, ::std::experimental::default_accessor<double> > fs_tensor{ };
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          LINALG_DETAIL::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const auto& const_fs_tensor( fs_tensor );
    auto submatrix = LINALG::submatrix( const_fs_tensor, 0, ::std::full_extent, ::std::tuple(0,1) );
    
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 0, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 0, 0, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 1, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 0, 1, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 2, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 0, 2, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 3, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 0, 3, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 4, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 0, 4, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 0, 1 ) ), ( LINALG_DETAIL::access( fs_tensor, 0, 0, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 1, 1 ) ), ( LINALG_DETAIL::access( fs_tensor, 0, 1, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 2, 1 ) ), ( LINALG_DETAIL::access( fs_tensor, 0, 2, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 3, 1 ) ), ( LINALG_DETAIL::access( fs_tensor ,0, 3, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 4, 1 ) ), ( LINALG_DETAIL::access( fs_tensor, 0, 4, 1 ) ) );
  }

  TEST( FS_TENSOR, CONST_SUBTENSOR )
  {
    using fs_tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 5, 5, 5 >, ::std::experimental::layout_right, ::std::experimental::default_accessor<double> >;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          LINALG_DETAIL::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const fs_tensor_type& const_fs_tensor( fs_tensor );
    auto subtensor = LINALG::subtensor( const_fs_tensor, ::std::tuple(2,5), ::std::tuple(2,4), ::std::tuple(2,3) );
    
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 0, 0, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 2, 2, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 1, 0, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 3, 2, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 2, 0, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 4, 2, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 0, 1, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 2, 3, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 1, 1, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 3, 3, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 2, 1, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 4, 3, 2 ) ) );
  }

  TEST( FS_TENSOR, SUBVECTOR )
  {
    // Construct
    LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 5, 5, 5 >, ::std::experimental::layout_right, ::std::experimental::default_accessor<double> > fs_tensor{ };
    // Set values in tensor
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          LINALG_DETAIL::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    // Get subvector
    auto subvector = LINALG::subvector( fs_tensor, 1, std::full_extent, 0 );
    // Modify view
    for ( auto i : { 1, 2, 3 } )
    {
      LINALG_DETAIL::access( subvector, i ) = val;
      val = 2 * val;
    }
    // Assert original tensor has been modified as well
    EXPECT_EQ( ( LINALG_DETAIL::access( subvector, 1 ) ), ( LINALG_DETAIL::access( fs_tensor, 1, 1, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subvector, 2 ) ), ( LINALG_DETAIL::access( fs_tensor, 1, 2, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subvector, 3 ) ), ( LINALG_DETAIL::access( fs_tensor ,1, 3, 0 ) ) );
  }

  TEST( FS_TENSOR, SUBMATRIX )
  {
    // Construct
    LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 5, 5, 5 >, ::std::experimental::layout_right, ::std::experimental::default_accessor<double> > fs_tensor{ };
    // Set values in tensor
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          LINALG_DETAIL::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    // Get submatrix
    auto submatrix = LINALG::submatrix( fs_tensor, 1, ::std::full_extent, ::std::tuple( 1, 4 ) );
    // Modify view
    for ( auto i : { 1, 2, 3 } )
    {
      for ( auto j : { 0, 1 } )
      {
        LINALG_DETAIL::access( submatrix, i, j ) = val;
        val = 2 * val;
      }
    }
    // Assert original tensor has been modified as well
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 1, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 1, 1, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 2, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 1, 2, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 3, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 1, 3, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 1, 1 ) ), ( LINALG_DETAIL::access( fs_tensor, 1, 1, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 2, 1 ) ), ( LINALG_DETAIL::access( fs_tensor, 1, 2, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 3, 1 ) ), ( LINALG_DETAIL::access( fs_tensor, 1, 3, 2 ) ) );
  }

  TEST( FS_TENSOR, SUBTENSOR )
  {
    using fs_tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 5, 5, 5 >, ::std::experimental::layout_right, ::std::experimental::default_accessor<double> >;
    // Default construct
    fs_tensor_type fs_tensor;
    // Set values in tensor tensor
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          LINALG_DETAIL::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    // Get subtensor
    auto subtensor = LINALG::subtensor( fs_tensor, ::std::tuple(2,5), ::std::tuple(2,4), ::std::tuple(2,3) );
    // Modify view
    for ( auto i : { 0, 1, 2 } )
    {
      for ( auto j : { 0, 1 } )
      {
        for ( auto k : { 0 } )
        {
          LINALG_DETAIL::access( subtensor, i,j, k ) = val;
          val = 2 * val;
        }
      }
    }
    // Assert original tensor has been modified as well
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 0, 0, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 2, 2, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 1, 0, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 3, 2, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 2, 0, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 4, 2, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 0, 1, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 2, 3, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 1, 1, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 3, 3, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 2, 1, 0 ) ), ( LINALG_DETAIL::access( fs_tensor, 4, 3, 2 ) ) );
  }

/*
  TEST( FS_TENSOR, NEGATION )
  {
    using tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Negate the tensor
    tensor_type negate_tensor { -tensor };
    // Access elements from const tensor
    auto val1 = std::experimental::math::detail::access( negate_tensor, 0, 0, 0 );
    auto val2 = std::experimental::math::detail::access( negate_tensor, 0, 0, 1 );
    auto val3 = std::experimental::math::detail::access( negate_tensor, 0, 1, 0 );
    auto val4 = std::experimental::math::detail::access( negate_tensor, 0, 1, 1 );
    auto val5 = std::experimental::math::detail::access( negate_tensor, 1, 0, 0 );
    auto val6 = std::experimental::math::detail::access( negate_tensor, 1, 0, 1 );
    auto val7 = std::experimental::math::detail::access( negate_tensor, 1, 1, 0 );
    auto val8 = std::experimental::math::detail::access( negate_tensor, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, -1.0 );
    EXPECT_EQ( val2, -2.0 );
    EXPECT_EQ( val3, -3.0 );
    EXPECT_EQ( val4, -4.0 );
    EXPECT_EQ( val5, -5.0 );
    EXPECT_EQ( val6, -6.0 );
    EXPECT_EQ( val7, -7.0 );
    EXPECT_EQ( val8, -8.0 );
  }

  TEST( FS_TENSOR, ADD )
  {
    using tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Add the two tensors together
    tensor_type tensor_sum { tensor + tensor_copy };
    // Access elements from const tensor
    auto val1 = std::experimental::math::detail::access( tensor_sum, 0, 0, 0 );
    auto val2 = std::experimental::math::detail::access( tensor_sum, 0, 0, 1 );
    auto val3 = std::experimental::math::detail::access( tensor_sum, 0, 1, 0 );
    auto val4 = std::experimental::math::detail::access( tensor_sum, 0, 1, 1 );
    auto val5 = std::experimental::math::detail::access( tensor_sum, 1, 0, 0 );
    auto val6 = std::experimental::math::detail::access( tensor_sum, 1, 0, 1 );
    auto val7 = std::experimental::math::detail::access( tensor_sum, 1, 1, 0 );
    auto val8 = std::experimental::math::detail::access( tensor_sum, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( FS_TENSOR, ADD_ASSIGN )
  {
    using tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Add the two tensors together
    static_cast<void>( tensor += tensor_copy );
    // Access elements from tensor
    auto val1 = std::experimental::math::detail::access( tensor, 0, 0, 0 );
    auto val2 = std::experimental::math::detail::access( tensor, 0, 0, 1 );
    auto val3 = std::experimental::math::detail::access( tensor, 0, 1, 0 );
    auto val4 = std::experimental::math::detail::access( tensor, 0, 1, 1 );
    auto val5 = std::experimental::math::detail::access( tensor, 1, 0, 0 );
    auto val6 = std::experimental::math::detail::access( tensor, 1, 0, 1 );
    auto val7 = std::experimental::math::detail::access( tensor, 1, 1, 0 );
    auto val8 = std::experimental::math::detail::access( tensor, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( FS_TENSOR, SUBTRACT )
  {
    using tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Subtract the two tensors
    tensor_type tensor_diff { tensor - tensor_copy };
    // Access elements from const tensor
    auto val1 = std::experimental::math::detail::access( tensor_diff, 0, 0, 0 );
    auto val2 = std::experimental::math::detail::access( tensor_diff, 0, 0, 1 );
    auto val3 = std::experimental::math::detail::access( tensor_diff, 0, 1, 0 );
    auto val4 = std::experimental::math::detail::access( tensor_diff, 0, 1, 1 );
    auto val5 = std::experimental::math::detail::access( tensor_diff, 1, 0, 0 );
    auto val6 = std::experimental::math::detail::access( tensor_diff, 1, 0, 1 );
    auto val7 = std::experimental::math::detail::access( tensor_diff, 1, 1, 0 );
    auto val8 = std::experimental::math::detail::access( tensor_diff, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0 );
    EXPECT_EQ( val2, 0 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 0 );
    EXPECT_EQ( val5, 0 );
    EXPECT_EQ( val6, 0 );
    EXPECT_EQ( val7, 0 );
    EXPECT_EQ( val8, 0 );
  }

  TEST( FS_TENSOR, SUBTRACT_ASSIGN )
  {
    using tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Subtract the two tensors
    static_cast<void>( tensor -= tensor_copy );
    // Access elements from tensor
    auto val1 = std::experimental::math::detail::access( tensor, 0, 0, 0 );
    auto val2 = std::experimental::math::detail::access( tensor, 0, 0, 1 );
    auto val3 = std::experimental::math::detail::access( tensor, 0, 1, 0 );
    auto val4 = std::experimental::math::detail::access( tensor, 0, 1, 1 );
    auto val5 = std::experimental::math::detail::access( tensor, 1, 0, 0 );
    auto val6 = std::experimental::math::detail::access( tensor, 1, 0, 1 );
    auto val7 = std::experimental::math::detail::access( tensor, 1, 1, 0 );
    auto val8 = std::experimental::math::detail::access( tensor, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0 );
    EXPECT_EQ( val2, 0 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 0 );
    EXPECT_EQ( val5, 0 );
    EXPECT_EQ( val6, 0 );
    EXPECT_EQ( val7, 0 );
    EXPECT_EQ( val8, 0 );
  }

  TEST( FS_TENSOR, SCALAR_PREMULTIPLY )
  {
    using tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Pre multiply
    tensor_type tensor_prod { 2 * tensor };
    // Access elements from const tensor
    auto val1 = std::experimental::math::detail::access( tensor_prod, 0, 0, 0 );
    auto val2 = std::experimental::math::detail::access( tensor_prod, 0, 0, 1 );
    auto val3 = std::experimental::math::detail::access( tensor_prod, 0, 1, 0 );
    auto val4 = std::experimental::math::detail::access( tensor_prod, 0, 1, 1 );
    auto val5 = std::experimental::math::detail::access( tensor_prod, 1, 0, 0 );
    auto val6 = std::experimental::math::detail::access( tensor_prod, 1, 0, 1 );
    auto val7 = std::experimental::math::detail::access( tensor_prod, 1, 1, 0 );
    auto val8 = std::experimental::math::detail::access( tensor_prod, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( FS_TENSOR, SCALAR_POSTMULTIPLY )
  {
    using tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Post multiply
    tensor_type tensor_prod { tensor * 2 };
    // Access elements from const tensor
    auto val1 = std::experimental::math::detail::access( tensor_prod, 0,0, 0 );
    auto val2 = std::experimental::math::detail::access( tensor_prod, 0,0, 1 );
    auto val3 = std::experimental::math::detail::access( tensor_prod, 0,1, 0 );
    auto val4 = std::experimental::math::detail::access( tensor_prod, 0,1, 1 );
    auto val5 = std::experimental::math::detail::access( tensor_prod, 1,0, 0 );
    auto val6 = std::experimental::math::detail::access( tensor_prod, 1,0, 1 );
    auto val7 = std::experimental::math::detail::access( tensor_prod, 1,1, 0 );
    auto val8 = std::experimental::math::detail::access( tensor_prod, 1,1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( FS_TENSOR, SCALAR_MULTIPLY_ASSIGN )
  {
    using tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Post multiply
    static_cast<void>( tensor *= 2 );
    // Access elements from const tensor
    auto val1 = std::experimental::math::detail::access( tensor, 0, 0, 0 );
    auto val2 = std::experimental::math::detail::access( tensor, 0, 0, 1 );
    auto val3 = std::experimental::math::detail::access( tensor, 0, 1, 0 );
    auto val4 = std::experimental::math::detail::access( tensor, 0, 1, 1 );
    auto val5 = std::experimental::math::detail::access( tensor, 1, 0, 0 );
    auto val6 = std::experimental::math::detail::access( tensor, 1, 0, 1 );
    auto val7 = std::experimental::math::detail::access( tensor, 1, 1, 0 );
    auto val8 = std::experimental::math::detail::access( tensor, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( FS_TENSOR, SCALAR_DIVIDE )
  {
    using tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Divide
    tensor_type tensor_divide { tensor / 2 };
    // Access elements from const tensor
    auto val1 = std::experimental::math::detail::access( tensor_divide, 0, 0, 0 );
    auto val2 = std::experimental::math::detail::access( tensor_divide, 0, 0, 1 );
    auto val3 = std::experimental::math::detail::access( tensor_divide, 0, 1, 0 );
    auto val4 = std::experimental::math::detail::access( tensor_divide, 0, 1, 1 );
    auto val5 = std::experimental::math::detail::access( tensor_divide, 1, 0, 0 );
    auto val6 = std::experimental::math::detail::access( tensor_divide, 1, 0, 1 );
    auto val7 = std::experimental::math::detail::access( tensor_divide, 1, 1, 0 );
    auto val8 = std::experimental::math::detail::access( tensor_divide, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0.5 );
    EXPECT_EQ( val2, 1.0 );
    EXPECT_EQ( val3, 1.5 );
    EXPECT_EQ( val4, 2.0 );
    EXPECT_EQ( val5, 2.5 );
    EXPECT_EQ( val6, 3.0 );
    EXPECT_EQ( val7, 3.5 );
    EXPECT_EQ( val8, 4.0 );
  }

  TEST( FS_TENSOR, SCALAR_DIVIDE_ASSIGN )
  {
    using tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::experimental::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::experimental::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::experimental::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::experimental::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::experimental::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::experimental::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::experimental::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::experimental::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Divide
    static_cast<void>( tensor /= 2 );
    // Access elements from const tensor
    auto val1 = std::experimental::math::detail::access( tensor, 0, 0, 0 );
    auto val2 = std::experimental::math::detail::access( tensor, 0, 0, 1 );
    auto val3 = std::experimental::math::detail::access( tensor, 0, 1, 0 );
    auto val4 = std::experimental::math::detail::access( tensor, 0, 1, 1 );
    auto val5 = std::experimental::math::detail::access( tensor, 1, 0, 0 );
    auto val6 = std::experimental::math::detail::access( tensor, 1, 0, 1 );
    auto val7 = std::experimental::math::detail::access( tensor, 1, 1, 0 );
    auto val8 = std::experimental::math::detail::access( tensor, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0.5 );
    EXPECT_EQ( val2, 1.0 );
    EXPECT_EQ( val3, 1.5 );
    EXPECT_EQ( val4, 2.0 );
    EXPECT_EQ( val5, 2.5 );
    EXPECT_EQ( val6, 3.0 );
    EXPECT_EQ( val7, 3.5 );
    EXPECT_EQ( val8, 4.0 );
  }

  TEST( TENSOR_VIEW, SIZE_AND_CAPACITY )
  {
    // Get a rank 3 subtensor
    using fs_tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5>;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::experimental::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const fs_tensor_type& const_fs_tensor( fs_tensor );
    auto subtensor = const_fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,4) );
    // Verify proper size and capacity
    EXPECT_TRUE( ( subtensor.size().extent(0) == 3 ) );
    EXPECT_TRUE( ( subtensor.size().extent(1) == 2 ) );
    EXPECT_TRUE( ( subtensor.size().extent(2) == 2 ) );
    EXPECT_TRUE( ( subtensor.capacity().extent(0) == 3 ) );
    EXPECT_TRUE( ( subtensor.capacity().extent(1) == 2 ) );
    EXPECT_TRUE( ( subtensor.capacity().extent(2) == 2 ) );
  }

  TEST( TENSOR_VIEW, CONST_SUBVECTOR )
  {
    // Get a rank 3 subtensor
    using fs_tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5>;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::experimental::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const fs_tensor_type& const_fs_tensor( fs_tensor );
    auto subtensor = const_fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,4) );
    auto subvector = subtensor.subvector( 0, std::experimental::full_extent, 1 );
    
    EXPECT_EQ( ( std::experimental::math::detail::access( subvector, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subvector, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 3, 3 ) ) );
  }

  TEST( TENSOR_VIEW, CONST_SUBMATRIX )
  {
    // Get a rank 3 subtensor
    using fs_tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5>;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::experimental::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const fs_tensor_type& const_fs_tensor( fs_tensor );
    auto subtensor = const_fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,4) );
    auto submatrix = subtensor.submatrix( 0, std::experimental::full_extent, std::experimental::full_extent );
    
    EXPECT_EQ( ( std::experimental::math::detail::access( submatrix, 0, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 2, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( submatrix, 0, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( submatrix, 1, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 3, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( submatrix, 1, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 3, 3 ) ) );
  }

  TEST( TENSOR_VIEW, CONST_SUBTENSOR )
  {
    // Get a rank 3 subtensor
    using fs_tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5>;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::experimental::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const fs_tensor_type& const_fs_tensor( fs_tensor );
    auto subtensor  = const_fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,4) );
    auto subtensor2 = subtensor.subtensor( std::tuple(0,2), std::experimental::full_extent, std::experimental::full_extent );
    
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor2, 0, 0, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 2, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor2, 0, 0, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor2, 0, 1, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 3, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor2, 0, 1, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 3, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor2, 1, 0, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 2, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor2, 1, 0, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor2, 1, 1, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 3, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor2, 1, 1, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 3, 3 ) ) );
  }

  TEST( TENSOR_VIEW, SUBVECTOR )
  {
    // Get a rank 3 subtensor
    using fs_tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5>;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::experimental::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    auto subtensor = fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,4) );
    auto subvector = subtensor.subvector( 0, std::experimental::full_extent, 1 );
    
    // Modify view
    for ( auto i : { 0, 1 } )
    {
      std::experimental::math::detail::access( subvector, i ) = val;
      val = 2 * val;
    }
    // Assert original tensor has been modified as well
    EXPECT_EQ( ( std::experimental::math::detail::access( subvector, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subvector, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 3, 3 ) ) );
  }

  TEST( TENSOR_VIEW, SUBMATRIX )
  {
    // Get a rank 3 subtensor
    using fs_tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5>;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::experimental::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    auto subtensor = fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,4) );
    auto submatrix = subtensor.submatrix( 0, std::experimental::full_extent, std::experimental::full_extent );
    
    // Modify view
    for ( auto i : { 0, 1 } )
    {
      for ( auto j : { 0, 1 } )
      {
        std::experimental::math::detail::access( submatrix, i, j ) = val;
        val = 2 * val;
      }
    }
    // Assert original tensor has been modified as well
    EXPECT_EQ( ( std::experimental::math::detail::access( submatrix, 0, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 2, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( submatrix, 0, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( submatrix, 1, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 3, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( submatrix, 1, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 3, 3 ) ) );
  }

  TEST( TENSOR_VIEW, SUBTENSOR )
  {
    // Get a rank 3 subtensor
    using fs_tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5>;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::experimental::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    auto subtensor = fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,4) );
    auto subtensor2 = subtensor.subtensor( std::tuple(0,2), std::experimental::full_extent, std::experimental::full_extent );
    
    // Modify view
    for ( auto i : { 0, 1 } )
    {
      for ( auto j : { 0, 1 } )
      {
        for ( auto k : { 0, 1 } )
        {
          std::experimental::math::detail::access( subtensor2, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    // Assert original tensor has been modified as well
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor2, 0, 0, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 2, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor2, 0, 0, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor2, 0, 1, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 3, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor2, 0, 1, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 3, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor2, 1, 0, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 2, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor2, 1, 0, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor2, 1, 1, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 3, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor2, 1, 1, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 3, 3 ) ) );
  }

  TEST( TENSOR_VIEW, NEGATION )
  {
    // Get a rank 3 subtensor
    using fs_tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5>;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::experimental::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const fs_tensor_type& const_fs_tensor( fs_tensor );
    auto subtensor = const_fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,4) );
    // Negate subtensor
    auto negate_subtensor = -subtensor;

    EXPECT_EQ( ( std::experimental::math::detail::access( negate_subtensor, 0, 0, 0 ) ), ( -std::experimental::math::detail::access( subtensor, 0, 0, 0 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( negate_subtensor, 1, 0, 0 ) ), ( -std::experimental::math::detail::access( subtensor, 1, 0, 0 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( negate_subtensor, 2, 0, 0 ) ), ( -std::experimental::math::detail::access( subtensor, 2, 0, 0 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( negate_subtensor, 0, 1, 0 ) ), ( -std::experimental::math::detail::access( subtensor, 0, 1, 0 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( negate_subtensor, 1, 1, 0 ) ), ( -std::experimental::math::detail::access( subtensor, 1, 1, 0 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( negate_subtensor, 2, 1, 0 ) ), ( -std::experimental::math::detail::access( subtensor, 2, 1, 0 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( negate_subtensor, 0, 0, 1 ) ), ( -std::experimental::math::detail::access( subtensor, 0, 0, 1 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( negate_subtensor, 1, 0, 1 ) ), ( -std::experimental::math::detail::access( subtensor, 1, 0, 1 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( negate_subtensor, 2, 0, 1 ) ), ( -std::experimental::math::detail::access( subtensor, 2, 0, 1 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( negate_subtensor, 0, 1, 1 ) ), ( -std::experimental::math::detail::access( subtensor, 0, 1, 1 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( negate_subtensor, 1, 1, 1 ) ), ( -std::experimental::math::detail::access( subtensor, 1, 1, 1 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( negate_subtensor, 2, 1, 1 ) ), ( -std::experimental::math::detail::access( subtensor, 2, 1, 1 ) ) );
  }

  TEST( TENSOR_VIEW, ADD )
  {
    // Get a rank 3 subtensor
    using fs_tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5>;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::experimental::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const fs_tensor_type& const_fs_tensor( fs_tensor );
    auto subtensor = const_fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,4) );
    // Add the subtensor with itself
    auto subtensor_sum = subtensor + subtensor;

    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_sum, 0, 0, 0 ) ), ( 2.0 * std::experimental::math::detail::access( const_fs_tensor, 2, 2, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_sum, 1, 0, 0 ) ), ( 2.0 * std::experimental::math::detail::access( const_fs_tensor, 3, 2, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_sum, 2, 0, 0 ) ), ( 2.0 * std::experimental::math::detail::access( const_fs_tensor, 4, 2, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_sum, 0, 1, 0 ) ), ( 2.0 * std::experimental::math::detail::access( const_fs_tensor, 2, 3, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_sum, 1, 1, 0 ) ), ( 2.0 * std::experimental::math::detail::access( const_fs_tensor, 3, 3, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_sum, 2, 1, 0 ) ), ( 2.0 * std::experimental::math::detail::access( const_fs_tensor, 4, 3, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_sum, 0, 0, 1 ) ), ( 2.0 * std::experimental::math::detail::access( const_fs_tensor, 2, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_sum, 1, 0, 1 ) ), ( 2.0 * std::experimental::math::detail::access( const_fs_tensor, 3, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_sum, 2, 0, 1 ) ), ( 2.0 * std::experimental::math::detail::access( const_fs_tensor, 4, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_sum, 0, 1, 1 ) ), ( 2.0 * std::experimental::math::detail::access( const_fs_tensor, 2, 3, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_sum, 1, 1, 1 ) ), ( 2.0 * std::experimental::math::detail::access( const_fs_tensor, 3, 3, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_sum, 2, 1, 1 ) ), ( 2.0 * std::experimental::math::detail::access( const_fs_tensor, 4, 3, 3 ) ) );
  }

  TEST( TENSOR_VIEW, ADD_ASSIGN )
  {
    // Get a rank 3 subtensor
    using fs_tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5>;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::experimental::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    auto subtensor = fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,4) );
    // Add the subtensor with itself
    static_cast<void>( subtensor += subtensor );

    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 0, 0, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 2, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 1, 0, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 2, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 2, 0, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 4, 2, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 0, 1, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 3, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 1, 1, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 3, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 2, 1, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 4, 3, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 0, 0, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 1, 0, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 2, 0, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 4, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 0, 1, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 3, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 1, 1, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 3, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 2, 1, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 4, 3, 3 ) ) );
  }

  TEST( TENSOR_VIEW, SUBTRACT )
  {
    // Get a rank 3 subtensor
    using fs_tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5>;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::experimental::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const fs_tensor_type& const_fs_tensor( fs_tensor );
    auto subtensor = const_fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,4) );
    // Subtract the subtensor with itself
    auto subtensor_diff = subtensor - subtensor;

    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_diff, 0, 0, 0 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_diff, 1, 0, 0 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_diff, 2, 0, 0 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_diff, 0, 1, 0 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_diff, 1, 1, 0 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_diff, 2, 1, 0 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_diff, 0, 0, 1 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_diff, 1, 0, 1 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_diff, 2, 0, 1 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_diff, 0, 1, 1 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_diff, 1, 1, 1 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_diff, 2, 1, 1 ) ), 0 );
  }

  TEST( TENSOR_VIEW, SUBTRACT_ASSIGN )
  {
    // Get a rank 3 subtensor
    using fs_tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5>;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::experimental::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    auto subtensor = fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,4) );
    // Subtract the subtensor with itself
    static_cast<void>( subtensor -= fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,4) ) );

    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 0, 0, 0 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 1, 0, 0 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 2, 0, 0 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 0, 1, 0 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 1, 1, 0 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 2, 1, 0 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 0, 0, 1 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 1, 0, 1 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 2, 0, 1 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 0, 1, 1 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 1, 1, 1 ) ), 0 );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 2, 1, 1 ) ), 0 );
  }

  TEST( TENSOR_VIEW, SCALAR_PREMULTIPLY )
  {
    // Get a rank 3 subtensor
    using fs_tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5>;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::experimental::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const fs_tensor_type& const_fs_tensor( fs_tensor );
    auto subtensor = const_fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,4) );
    // Multiply the subtensor with a constant
    auto subtensor_prod = 2.0 * subtensor;

    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 0, 0, 0 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 0, 0, 0 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 1, 0, 0 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 1, 0, 0 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 2, 0, 0 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 2, 0, 0 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 0, 1, 0 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 0, 1, 0 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 1, 1, 0 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 1, 1, 0 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 2, 1, 0 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 2, 1, 0 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 0, 0, 1 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 0, 0, 1 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 1, 0, 1 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 1, 0, 1 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 2, 0, 1 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 2, 0, 1 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 0, 1, 1 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 0, 1, 1 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 1, 1, 1 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 1, 1, 1 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 2, 1, 1 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 2, 1, 1 ) ) );
  }

  TEST( TENSOR_VIEW, SCALAR_POSTMULTIPLY )
  {
    // Get a rank 3 subtensor
    using fs_tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5>;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::experimental::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const fs_tensor_type& const_fs_tensor( fs_tensor );
    auto subtensor = const_fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,4) );
    // Multiply the subtensor with a constant
    auto subtensor_prod = subtensor * 2.0;

    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 0, 0, 0 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 0, 0, 0 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 1, 0, 0 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 1, 0, 0 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 2, 0, 0 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 2, 0, 0 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 0, 1, 0 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 0, 1, 0 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 1, 1, 0 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 1, 1, 0 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 2, 1, 0 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 2, 1, 0 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 0, 0, 1 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 0, 0, 1 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 1, 0, 1 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 1, 0, 1 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 2, 0, 1 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 2, 0, 1 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 0, 1, 1 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 0, 1, 1 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 1, 1, 1 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 1, 1, 1 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_prod, 2, 1, 1 ) ), ( 2.0 * std::experimental::math::detail::access( subtensor, 2, 1, 1 ) ) );
  }

  TEST( TENSOR_VIEW, SCALAR_MULTIPLY_ASSIGN )
  {
    // Get a rank 3 subtensor
    using fs_tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5>;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::experimental::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    auto subtensor = fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,4) );
    // Multiply the subtensor with a constant
    static_cast<void>( subtensor *= 2.0 );

    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 0, 0, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 2, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 1, 0, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 2, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 2, 0, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 4, 2, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 0, 1, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 3, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 1, 1, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 3, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 2, 1, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 4, 3, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 0, 0, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 1, 0, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 2, 0, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 4, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 0, 1, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 3, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 1, 1, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 3, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 2, 1, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 4, 3, 3 ) ) );
  }

  TEST( TENSOR_VIEW, SCALAR_DIVIDE )
  {
    // Get a rank 3 subtensor
    using fs_tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5>;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::experimental::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const fs_tensor_type& const_fs_tensor( fs_tensor );
    auto subtensor = const_fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,4) );
    // Divide the subtensor with a constant
    auto subtensor_divide = subtensor / 2.0;

    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_divide, 0, 0, 0 ) ), ( std::experimental::math::detail::access( subtensor, 0, 0, 0 ) / 2.0 ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_divide, 1, 0, 0 ) ), ( std::experimental::math::detail::access( subtensor, 1, 0, 0 ) / 2.0 ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_divide, 2, 0, 0 ) ), ( std::experimental::math::detail::access( subtensor, 2, 0, 0 ) / 2.0 ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_divide, 0, 1, 0 ) ), ( std::experimental::math::detail::access( subtensor, 0, 1, 0 ) / 2.0 ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_divide, 1, 1, 0 ) ), ( std::experimental::math::detail::access( subtensor, 1, 1, 0 ) / 2.0 ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_divide, 2, 1, 0 ) ), ( std::experimental::math::detail::access( subtensor, 2, 1, 0 ) / 2.0 ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_divide, 0, 0, 1 ) ), ( std::experimental::math::detail::access( subtensor, 0, 0, 1 ) / 2.0 ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_divide, 1, 0, 1 ) ), ( std::experimental::math::detail::access( subtensor, 1, 0, 1 ) / 2.0 ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_divide, 2, 0, 1 ) ), ( std::experimental::math::detail::access( subtensor, 2, 0, 1 ) / 2.0 ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_divide, 0, 1, 1 ) ), ( std::experimental::math::detail::access( subtensor, 0, 1, 1 ) / 2.0 ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_divide, 1, 1, 1 ) ), ( std::experimental::math::detail::access( subtensor, 1, 1, 1 ) / 2.0 ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor_divide, 2, 1, 1 ) ), ( std::experimental::math::detail::access( subtensor, 2, 1, 1 ) / 2.0 ) );
  }

  TEST( TENSOR_VIEW, SCALAR_DIVIDE_ASSIGN )
  {
    // Get a rank 3 subtensor
    using fs_tensor_type = std::experimental::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5>;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::experimental::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    auto subtensor = fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,4) );
    // Divide the subtensor with a constant
    static_cast<void>( subtensor /= 2.0 );

    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 0, 0, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 2, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 1, 0, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 2, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 2, 0, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 4, 2, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 0, 1, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 3, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 1, 1, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 3, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 2, 1, 0 ) ), ( std::experimental::math::detail::access( fs_tensor, 4, 3, 2 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 0, 0, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 1, 0, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 2, 0, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 4, 2, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 0, 1, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 2, 3, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 1, 1, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 3, 3, 3 ) ) );
    EXPECT_EQ( ( std::experimental::math::detail::access( subtensor, 2, 1, 1 ) ), ( std::experimental::math::detail::access( fs_tensor, 4, 3, 3 ) ) );
  }
*/
}