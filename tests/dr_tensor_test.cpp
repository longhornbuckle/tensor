#include <gtest/gtest.h>
#include <experimental/linear_algebra.hpp>

namespace
{
  TEST( DR_TENSOR, DEFAULT_CONSTRUCTOR_AND_DESTRUCTOR )
  {
    // Default construction
    [[maybe_unused]] LINALG::dyn_tensor< double, 2 > dyn_tensor;
    // Destructor will be called when unit test ends and the dr tensor exits scope
  }

  TEST( DR_TENSOR, MUTABLE_AND_CONST_INDEX_ACCESS )
  {
    // Construct
    LINALG::dyn_tensor< double, 3 >
      dyn_tensor { ::std::extents< ::std::size_t, 2, 2, 2 > {} };
    // Populate via mutable index access
    LINALG_DETAIL::access( dyn_tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 1, 1 ) = 8.0;
    // Get a const reference
    const LINALG::dr_tensor< double, ::std::extents< ::std::size_t, ::std::dynamic_extent, ::std::dynamic_extent, ::std::dynamic_extent > >&
      const_dyn_tensor( dyn_tensor );
    // Access elements from const fs tensor
    auto val1 = LINALG_DETAIL::access( const_dyn_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( const_dyn_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( const_dyn_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( const_dyn_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( const_dyn_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( const_dyn_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( const_dyn_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( const_dyn_tensor, 1, 1, 1 );
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

  TEST( DR_TENSOR, COPY_CONSTRUCTOR )
  {
    // Construct
    LINALG::dyn_tensor< double, 3 > dyn_tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( dyn_tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    LINALG::dyn_tensor< double, 3 > dyn_tensor_copy{ dyn_tensor };
    // Get a const reference to copy
    const LINALG::dyn_tensor< double, 3 >& const_dyn_tensor( dyn_tensor_copy );
    // Access elements from const dyn tensor
    auto val1 = LINALG_DETAIL::access( const_dyn_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( const_dyn_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( const_dyn_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( const_dyn_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( const_dyn_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( const_dyn_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( const_dyn_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( const_dyn_tensor, 1, 1, 1 );
    // Check the dyn tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( DR_TENSOR, MOVE_CONSTRUCTOR )
  {
    // Construct
    LINALG::dyn_tensor< double, 3 > dyn_tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( dyn_tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 1, 1 ) = 8.0;
    // Move construct
    LINALG::dyn_tensor< double, 3 > dyn_tensor_move{ ::std::move( dyn_tensor ) };
    // Get a const reference to moved tensor
    const LINALG::dyn_tensor< double, 3 >& const_dyn_tensor( dyn_tensor_move );
    // Access elements from const fs tensor
    auto val1 = LINALG_DETAIL::access( const_dyn_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( const_dyn_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( const_dyn_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( const_dyn_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( const_dyn_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( const_dyn_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( const_dyn_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( const_dyn_tensor, 1, 1, 1 );
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

  TEST( DR_TENSOR, CONSTRUCT_FROM_VIEW )
  {
    // Construct
    LINALG::dyn_tensor< double, 3 > dyn_tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( dyn_tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 1, 1 ) = 8.0;
    // Construct from view
    LINALG::dyn_tensor< double, 3 > const_dyn_tensor( ::std::mdspan( dyn_tensor.data_handle(), dyn_tensor.mapping(), dyn_tensor.accessor() ) );
    // Access elements from const dyn tensor
    auto val1 = LINALG_DETAIL::access( const_dyn_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( const_dyn_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( const_dyn_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( const_dyn_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( const_dyn_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( const_dyn_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( const_dyn_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( const_dyn_tensor, 1, 1, 1 );
    // Check the dr tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( DR_TENSOR, CONSTRUCT_FROM_FS_TENSOR )
  {
    using float_left_tensor_type   = LINALG::fs_tensor< float, ::std::extents< ::std::size_t, 2, 2, 2 >, std::experimental::layout_right,std::experimental::default_accessor<float> >;
    using double_right_tensor_type = LINALG::dyn_tensor< double, 3 >;
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
    double_right_tensor_type dyn_tensor_copy{ fs_tensor };
    // Access elements from const fs tensor
    auto val1 = LINALG_DETAIL::access( dyn_tensor_copy, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( dyn_tensor_copy, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( dyn_tensor_copy, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( dyn_tensor_copy, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( dyn_tensor_copy, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( dyn_tensor_copy, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( dyn_tensor_copy, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( dyn_tensor_copy, 1, 1, 1 );
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

  TEST( DR_TENSOR, ASSIGNMENT_OPERATOR )
  {
    // Construct
    LINALG::dyn_tensor< double, 3 > dyn_tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( dyn_tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 1, 1 ) = 8.0;
    // Default construct and assign
    LINALG::dyn_tensor< double, 3 > dyn_tensor_copy;
    dyn_tensor_copy = dyn_tensor;
    // Access elements from dyn tensor
    auto val1 = LINALG_DETAIL::access( dyn_tensor_copy, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( dyn_tensor_copy, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( dyn_tensor_copy, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( dyn_tensor_copy, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( dyn_tensor_copy, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( dyn_tensor_copy, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( dyn_tensor_copy, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( dyn_tensor_copy, 1, 1, 1 );
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

  TEST( DR_TENSOR, TENSOR_EXPRESSION_ASSIGNMENT_OPERATOR )
  {
    using float_left_tensor_type = LINALG::fs_tensor< float, ::std::extents< ::std::size_t, 2, 2, 2 >, ::std::layout_right, ::std::default_accessor<float> >;
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
    LINALG::dyn_tensor< double, 3 > dyn_tensor_copy;
    dyn_tensor_copy = fs_tensor;
    // Access elements from const fs tensor tensor
    auto val1 = LINALG_DETAIL::access( dyn_tensor_copy, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( dyn_tensor_copy, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( dyn_tensor_copy, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( dyn_tensor_copy, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( dyn_tensor_copy, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( dyn_tensor_copy, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( dyn_tensor_copy, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( dyn_tensor_copy, 1, 1, 1 );
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

  TEST( DR_TENSOR, ASSIGN_FROM_VIEW )
  {
    using fs_tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 >, ::std::layout_right, ::std::default_accessor<double> >;
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
    // Construct and assign from view
    auto dyn_tensor_view = ::std::mdspan( fs_tensor.data_handle(), fs_tensor.mapping(), fs_tensor.accessor() );
    // Access elements from const fs tensor
    auto val1 = LINALG_DETAIL::access( dyn_tensor_view, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( dyn_tensor_view, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( dyn_tensor_view, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( dyn_tensor_view, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( dyn_tensor_view, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( dyn_tensor_view, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( dyn_tensor_view, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( dyn_tensor_view, 1, 1, 1 );
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

  TEST( DR_TENSOR, SIZE_AND_CAPACITY )
  {
    // Construct
    LINALG::dyn_tensor< double, 4 > dyn_tensor{ ::std::extents< ::std::size_t, 2, 5, 1, 7 >() };
    EXPECT_TRUE( ( dyn_tensor.extents().extent(0) == 2 ) );
    EXPECT_TRUE( ( dyn_tensor.extents().extent(1) == 5 ) );
    EXPECT_TRUE( ( dyn_tensor.extents().extent(2) == 1 ) );
    EXPECT_TRUE( ( dyn_tensor.extents().extent(3) == 7 ) );
    dyn_tensor.reserve( ::std::extents< ::std::size_t, 3, 5, 2, 10 >() );
    EXPECT_TRUE( ( dyn_tensor.capacity().extent(0) == 3 ) );
    EXPECT_TRUE( ( dyn_tensor.capacity().extent(1) == 5 ) );
    EXPECT_TRUE( ( dyn_tensor.capacity().extent(2) == 2 ) );
    EXPECT_TRUE( ( dyn_tensor.capacity().extent(3) == 10 ) );
  }

  TEST( DR_TENSOR, RESIZE )
  {
    // Construct
    LINALG::dyn_tensor< double, 3 > dyn_tensor{ ::std::extents<::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( dyn_tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 1, 1 ) = 8.0;
    // Resize
    dyn_tensor.resize( ::std::extents< ::std::size_t, 3, 3, 3 >() );
    LINALG_DETAIL::access( dyn_tensor, 0, 0, 2 ) = 9.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 1, 2 ) = 10.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 2, 0 ) = 11.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 2, 1 ) = 12.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 2, 2 ) = 13.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 0, 2 ) = 14.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 1, 2 ) = 15.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 2, 0 ) = 16.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 2, 1 ) = 17.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 2, 2 ) = 18.0;
    LINALG_DETAIL::access( dyn_tensor, 2, 0, 0 ) = 19.0;
    LINALG_DETAIL::access( dyn_tensor, 2, 0, 1 ) = 20.0;
    LINALG_DETAIL::access( dyn_tensor, 2, 0, 2 ) = 21.0;
    LINALG_DETAIL::access( dyn_tensor, 2, 1, 0 ) = 22.0;
    LINALG_DETAIL::access( dyn_tensor, 2, 1, 1 ) = 23.0;
    LINALG_DETAIL::access( dyn_tensor, 2, 1, 2 ) = 24.0;
    LINALG_DETAIL::access( dyn_tensor, 2, 2, 0 ) = 25.0;
    LINALG_DETAIL::access( dyn_tensor, 2, 2, 1 ) = 26.0;
    LINALG_DETAIL::access( dyn_tensor, 2, 2, 2 ) = 27.0;
    // Get values
    auto val1  = LINALG_DETAIL::access( dyn_tensor, 0, 0, 0 );
    auto val2  = LINALG_DETAIL::access( dyn_tensor, 0, 0, 1 );
    auto val3  = LINALG_DETAIL::access( dyn_tensor, 0, 1, 0 );
    auto val4  = LINALG_DETAIL::access( dyn_tensor, 0, 1, 1 );
    auto val5  = LINALG_DETAIL::access( dyn_tensor, 1, 0, 0 );
    auto val6  = LINALG_DETAIL::access( dyn_tensor, 1, 0, 1 );
    auto val7  = LINALG_DETAIL::access( dyn_tensor, 1, 1, 0 );
    auto val8  = LINALG_DETAIL::access( dyn_tensor, 1, 1, 1 );
    auto val9  = LINALG_DETAIL::access( dyn_tensor, 0, 0, 2 );
    auto val10 = LINALG_DETAIL::access( dyn_tensor, 0, 1, 2 );
    auto val11 = LINALG_DETAIL::access( dyn_tensor, 0, 2, 0 );
    auto val12 = LINALG_DETAIL::access( dyn_tensor, 0, 2, 1 );
    auto val13 = LINALG_DETAIL::access( dyn_tensor, 0, 2, 2 );
    auto val14 = LINALG_DETAIL::access( dyn_tensor, 1, 0, 2 );
    auto val15 = LINALG_DETAIL::access( dyn_tensor, 1, 1, 2 );
    auto val16 = LINALG_DETAIL::access( dyn_tensor, 1, 2, 0 );
    auto val17 = LINALG_DETAIL::access( dyn_tensor, 1, 2, 1 );
    auto val18 = LINALG_DETAIL::access( dyn_tensor, 1, 2, 2 );
    auto val19 = LINALG_DETAIL::access( dyn_tensor, 2, 0, 0 );
    auto val20 = LINALG_DETAIL::access( dyn_tensor, 2, 0, 1 );
    auto val21 = LINALG_DETAIL::access( dyn_tensor, 2, 0, 2 );
    auto val22 = LINALG_DETAIL::access( dyn_tensor, 2, 1, 0 );
    auto val23 = LINALG_DETAIL::access( dyn_tensor, 2, 1, 1 );
    auto val24 = LINALG_DETAIL::access( dyn_tensor, 2, 1, 2 );
    auto val25 = LINALG_DETAIL::access( dyn_tensor, 2, 2, 0 );
    auto val26 = LINALG_DETAIL::access( dyn_tensor, 2, 2, 1 );
    auto val27 = LINALG_DETAIL::access( dyn_tensor, 2, 2, 2 );
    // Check the values are correct
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
    EXPECT_EQ( val9, 9.0 );
    EXPECT_EQ( val10, 10.0 );
    EXPECT_EQ( val11, 11.0 );
    EXPECT_EQ( val12, 12.0 );
    EXPECT_EQ( val13, 13.0 );
    EXPECT_EQ( val14, 14.0 );
    EXPECT_EQ( val15, 15.0 );
    EXPECT_EQ( val16, 16.0 );
    EXPECT_EQ( val17, 17.0 );
    EXPECT_EQ( val18, 18.0 );
    EXPECT_EQ( val19, 19.0 );
    EXPECT_EQ( val20, 20.0 );
    EXPECT_EQ( val21, 21.0 );
    EXPECT_EQ( val22, 22.0 );
    EXPECT_EQ( val23, 23.0 );
    EXPECT_EQ( val24, 24.0 );
    EXPECT_EQ( val25, 25.0 );
    EXPECT_EQ( val26, 26.0 );
    EXPECT_EQ( val27, 27.0 );
  }

  TEST( DR_TENSOR, RESERVE )
  {
    // Construct
    LINALG::dyn_tensor< double, 3 > dyn_tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( dyn_tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( dyn_tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( dyn_tensor, 1, 1, 1 ) = 8.0;
    // Resize
    dyn_tensor.reserve( ::std::extents<size_t,4,4,4>() );
    // Get values
    auto val1  = LINALG_DETAIL::access( dyn_tensor, 0, 0, 0 );
    auto val2  = LINALG_DETAIL::access( dyn_tensor, 0, 0, 1 );
    auto val3  = LINALG_DETAIL::access( dyn_tensor, 0, 1, 0 );
    auto val4  = LINALG_DETAIL::access( dyn_tensor, 0, 1, 1 );
    auto val5  = LINALG_DETAIL::access( dyn_tensor, 1, 0, 0 );
    auto val6  = LINALG_DETAIL::access( dyn_tensor, 1, 0, 1 );
    auto val7  = LINALG_DETAIL::access( dyn_tensor, 1, 1, 0 );
    auto val8  = LINALG_DETAIL::access( dyn_tensor, 1, 1, 1 );
    // Check the values are correct
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( DR_TENSOR, CONST_SUBVECTOR )
  {
    // Construct
    LINALG::dyn_tensor< double, 3 > dyn_tensor{ ::std::extents< ::std::size_t, 5, 5, 5 >() };
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          LINALG_DETAIL::access( dyn_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const LINALG::dyn_tensor< double, 3 >& const_dyn_tensor( dyn_tensor );
    auto subvector = LINALG::subvector( const_dyn_tensor, 0, ::std::full_extent, 1 );
    
    EXPECT_EQ( ( LINALG_DETAIL::access( subvector, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 0, 0, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subvector, 1 ) ), ( LINALG_DETAIL::access( dyn_tensor, 0, 1, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subvector, 2 ) ), ( LINALG_DETAIL::access( dyn_tensor, 0, 2, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subvector, 3 ) ), ( LINALG_DETAIL::access( dyn_tensor, 0, 3, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subvector, 4 ) ), ( LINALG_DETAIL::access( dyn_tensor, 0, 4, 1 ) ) );
  }

  TEST( DR_TENSOR, CONST_SUBMATRIX )
  {
    // Construct
    LINALG::dyn_tensor< double, 3 > dyn_tensor { ::std::extents< ::std::size_t, 5, 5, 5 >() };
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          LINALG_DETAIL::access( dyn_tensor, i,j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const LINALG::dyn_tensor< double, 3 >& const_dyn_tensor( dyn_tensor );
    auto submatrix = LINALG::submatrix( const_dyn_tensor, 0, ::std::full_extent, ::std::tuple( 0, 1 ) );
    
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 0, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 0, 0, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 1, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 0, 1, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 2, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 0, 2, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 3, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 0, 3, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 4, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 0, 4, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 0, 1 ) ), ( LINALG_DETAIL::access( dyn_tensor, 0, 0, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 1, 1 ) ), ( LINALG_DETAIL::access( dyn_tensor, 0, 1, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 2, 1 ) ), ( LINALG_DETAIL::access( dyn_tensor, 0, 2, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 3, 1 ) ), ( LINALG_DETAIL::access( dyn_tensor, 0, 3, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 4, 1 ) ), ( LINALG_DETAIL::access( dyn_tensor, 0, 4, 1 ) ) );
  }

  TEST( DR_TENSOR, CONST_SUBTENSOR )
  {
    // Construct
    LINALG::dyn_tensor< double, 3 > dyn_tensor{ ::std::extents< ::std::size_t, 5, 5, 5 >() };
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          LINALG_DETAIL::access( dyn_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const LINALG::dyn_tensor< double, 3 >& const_dyn_tensor( dyn_tensor );
    auto subtensor = LINALG::subtensor( const_dyn_tensor, ::std::tuple(2,5), ::std::tuple(2,4), ::std::tuple(2,3) );
    
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 0, 0, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 2, 2, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 1, 0, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 3, 2, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 2, 0, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 4, 2, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 0, 1, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 2, 3, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 1, 1, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 3, 3, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 2, 1, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 4, 3, 2 ) ) );
  }

  TEST( DR_TENSOR, SUBVECTOR )
  {
    // Construct
    LINALG::dyn_tensor< double, 3 > dyn_tensor{ ::std::extents< ::std::size_t, 5, 5, 5 >() };
    // Set values in tensor
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          LINALG_DETAIL::access( dyn_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    // Get subvector
    auto subvector = LINALG::subvector( dyn_tensor, 1, ::std::full_extent, 0 );
    // Modify view
    for ( auto i : { 1, 2, 3 } )
    {
      LINALG_DETAIL::access( subvector, i ) = val;
      val = 2 * val;
    }
    // Assert original tensor has been modified as well
    EXPECT_EQ( ( LINALG_DETAIL::access( subvector, 1 ) ), ( LINALG_DETAIL::access( dyn_tensor, 1, 1, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subvector, 2 ) ), ( LINALG_DETAIL::access( dyn_tensor, 1, 2, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subvector, 3 ) ), ( LINALG_DETAIL::access( dyn_tensor, 1, 3, 0 ) ) );
  }

  TEST( DR_TENSOR, SUBMATRIX )
  {
    // Construct
    LINALG::dyn_tensor< double, 3 > dyn_tensor{ ::std::extents< ::std::size_t, 5, 5, 5 >() };
    // Set values in tensor
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          LINALG_DETAIL::access( dyn_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    // Get submatrix
    auto submatrix = LINALG::submatrix( dyn_tensor, 1, ::std::full_extent, ::std::tuple( 1, 4 ) );
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
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 1, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 1, 1, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 2, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 1, 2, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 3, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 1, 3, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 1, 1 ) ), ( LINALG_DETAIL::access( dyn_tensor, 1, 1, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 2, 1 ) ), ( LINALG_DETAIL::access( dyn_tensor, 1, 2, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( submatrix, 3, 1 ) ), ( LINALG_DETAIL::access( dyn_tensor, 1, 3, 2 ) ) );
  }

  TEST( DR_TENSOR, SUBTENSOR )
  {
    // Construct
    LINALG::dyn_tensor< double, 3 > dyn_tensor{ ::std::extents< ::std::size_t, 5, 5, 5 >() };
    // Set values in tensor
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          LINALG_DETAIL::access( dyn_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    // Get subtensor
    auto subtensor = LINALG::subtensor( dyn_tensor, ::std::tuple(2,5), ::std::tuple(2,4), ::std::tuple(2,3) );
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
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 0, 0, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 2, 2, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 1, 0, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 3, 2, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 2, 0, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 4, 2, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 0, 1, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 2, 3, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 1, 1, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 3, 3, 2 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( subtensor, 2, 1, 0 ) ), ( LINALG_DETAIL::access( dyn_tensor, 4, 3, 2 ) ) );
  }
}