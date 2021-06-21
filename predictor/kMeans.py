import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from loguru import logger


def extract_sales_heat(
    sales_info: pd.DataFrame,
):
    N_CLUSTER = 3

    # date_block_num  shop_id  item_category_id  cat_shop_total_sales_sum  cat_shop_total_sales_mean
    final_data_list = []
    for i in range(60):
        logger.debug(f"Shop: {i}")
        for j in range(34):
            # Copy sales info
            km_train_data = sales_info.query(f"shop_id == {i} and date_block_num == {j}").copy()
            # print(km_train_data)

            # Remove duplicated value and Zero Value
            km_cluster_data = km_train_data.loc[~km_train_data["cat_shop_total_sales_sum"].duplicated()].query("cat_shop_total_sales_sum > 0")

            range_list = []
            if len(km_cluster_data) > 10:
                # Do Normalization for Clustering
                km_cluster_data["cat_shop_total_sales_sum_norm"] = km_cluster_data["cat_shop_total_sales_sum"].apply(lambda x: x ** 0.125)

                # Do Clustering
                km_cluster_data["cat_shop_sales_heat"] = KMeans(n_clusters=N_CLUSTER).fit_predict(km_cluster_data[["cat_shop_total_sales_sum_norm"]])
                # print(km_cluster_data[["total_sales_sum", "cat_shop_sales_heat"]])
                # print(km_cluster_data["cat_shop_sales_heat"].min(), km_cluster_data["cat_shop_sales_heat"].max())

                # Do Heat Adjustment by range - sort range
                for it in range(N_CLUSTER):
                    temp_mapper = km_cluster_data.query(f"cat_shop_sales_heat == {it}")
                    range_tuple = (
                        temp_mapper["cat_shop_total_sales_sum"].min(),
                        temp_mapper["cat_shop_total_sales_sum"].max(),
                        it + 1
                    )
                    range_list.append(range_tuple)

                # Join Final Result of Joined Data
                final_data = km_train_data.join(
                    km_cluster_data["cat_shop_sales_heat"],
                    how='left'
                ).copy()
                # print(final_data)

                # Label Zero Sales range info in range list
                final_data.loc[
                    final_data["cat_shop_total_sales_sum"] == 0,
                    "cat_shop_sales_heat"
                ] = 0

                for lb, ub, label_value in range_list:
                    # print(lb, ub, label_value)
                    final_data.loc[
                            (final_data["cat_shop_total_sales_sum"] >= lb)
                            & (final_data["cat_shop_total_sales_sum"] <= ub),
                            "cat_shop_sales_heat"
                    ] = label_value
                    # print(final_data.loc[
                    #             (final_data["cat_shop_total_sales_sum"] >= lb) & (final_data["cat_shop_total_sales_sum"] <= ub)
                    #     ]
                    # )

                # Sort List
                range_list.sort(key=lambda x: x[0])
                # print(range_list)

                # Re-mapping sales heat
                heat_mapper = {0: 0}
                for it in range(N_CLUSTER):
                    heat_mapper[
                        range_list[it][2]
                    ] = it + 1
                final_data["cat_shop_sales_heat"].replace(
                    heat_mapper,
                    inplace=True,
                )

                assert not final_data.isna().values.any(), "Final Data Includes NaN"
                assert len(final_data) == 84, "Length Error"
            elif len(km_cluster_data) > 0:
                # print(f"Shop: {i}, Date_Block_Num: {j}")
                # print(km_cluster_data)
                final_data = km_train_data.copy()
                final_data["cat_shop_sales_heat"] = pd.Series(
                    [0 for _ in range(len(km_train_data))],
                    index=km_train_data.index
                )

                final_data.loc[final_data["cat_shop_total_sales_sum"] > 0, "cat_shop_sales_heat"] = 1
                # print(final_data)
                assert len(final_data) == 84, "Length Error"
            else:
                final_data = km_train_data.copy()
                final_data["cat_shop_sales_heat"] = pd.Series(
                    [0 for _ in range(len(km_train_data))],
                    index=km_train_data.index
                )
                # print(final_data)


            final_data_list.append(final_data)
            # print(final_data)
            # print(final_data.isna().values.any())

    sales_info_with_heat = pd.concat(
        final_data_list,
        axis=0,
    ).sort_index()

    assert not sales_info_with_heat.isna().values.any(), "Info includes NaN"

    return sales_info_with_heat


def extract_monthly_item_sales_heat(
    input_data: pd.DataFrame
):
    #------------------------------------------------
    # date_block_num, item_id,
    # monthly_total_item_sales_sum,
    # monthly_total_item_sales_mean
    #------------------------------------------------
    N_CLUSTER = 5

    final_data_list = []
    for month_iter in range(34):
        logger.debug(f"Month: {month_iter}")

        # Extract Train Data
        km_train_data = input_data.query(f"date_block_num == {month_iter}").copy()
        # print(
        #     km_train_data["monthly_total_item_sales_sum"].min(),
        #     km_train_data["monthly_total_item_sales_sum"].max()
        # )

        # Remove duplicated value and Zero Value
        # TODO: Check Points for clustering
        km_cluster_data = (
            km_train_data.loc[
                ~km_train_data["monthly_total_item_sales_sum"].duplicated()
            ]
            .query("monthly_total_item_sales_sum > 0")
        )

        range_list = []
        # Do Normalization for Clustering
        km_cluster_data["monthly_total_item_sales_sum_norm"] = (
            km_cluster_data["monthly_total_item_sales_sum"].apply(lambda x: x ** 0.125)
        )

        # Do Clustering
        km_cluster_data["monthly_total_item_sales_heat"] = (
            KMeans(n_clusters=N_CLUSTER).fit_predict(
                km_cluster_data[["monthly_total_item_sales_sum_norm"]]
            )
        )

        # Do Heat Adjustment by range - add to list
        for it in range(N_CLUSTER):
            temp_mapper = km_cluster_data.query(f"monthly_total_item_sales_heat == {it}")
            # Don`t forget to denormalize
            range_tuple = (
                temp_mapper["monthly_total_item_sales_sum"].min(),
                temp_mapper["monthly_total_item_sales_sum"].max(),
                it + 1
            )
            range_list.append(range_tuple)

        # Join Final Result of Joined Data
        final_data = km_train_data.join(
            km_cluster_data["monthly_total_item_sales_heat"],
            how='left'
        ).copy()

        # Replace Heat Value with item sales sum is zero
        final_data.loc[
            final_data["monthly_total_item_sales_sum"] == 0.0,
            "monthly_total_item_sales_heat"
        ] = 0

        for lb, ub, label_value in range_list:
            # print(lb, ub, label_value)

            final_data.loc[
                (final_data["monthly_total_item_sales_sum"] >= lb)
              & (final_data["monthly_total_item_sales_sum"] <= ub),
                "monthly_total_item_sales_heat"
            ] = label_value

        # print(len(final_data))
        # print(final_data)

        # Sort List
        range_list.sort(key=lambda x: x[0])
        # print(range_list)

        # Re-mapping sales heat
        heat_mapper = {0: 0}
        for it in range(N_CLUSTER):
            heat_mapper[
                range_list[it][2]
            ] = it + 1
        final_data["monthly_total_item_sales_heat"].replace(
            heat_mapper,
            inplace=True
        )

        # print(final_data)
        assert not final_data.isna().values.any(), "Final Data Contains NaN"
        final_data_list.append(final_data)

    monthly_total_item_sales_with_heat = pd.concat(
        final_data_list,
        axis=0,
    ).sort_index()

    assert len(monthly_total_item_sales_with_heat) == 34*22170, "Length of output is incorrect"

    return monthly_total_item_sales_with_heat


def extract_monthly_item_cat_sales_heat(
    input_data: pd.DataFrame
):
    #------------------------------------------------
    # date_block_num, item_id,
    # monthly_total_item_cat_sales_sum,
    # monthly_total_item_cat_sales_mean
    #------------------------------------------------
    N_CLUSTER = 5

    final_data_list = []
    for month_iter in range(34):
        logger.debug(f"Month: {month_iter}")

        # Extract Train Data
        km_train_data = input_data.query(f"date_block_num == {month_iter}").copy()
        # print(
        #     km_train_data["monthly_total_item_sales_sum"].min(),
        #     km_train_data["monthly_total_item_sales_sum"].max()
        # )

        # Remove duplicated value and Zero Value
        # TODO: Check Points for clustering
        km_cluster_data = (
            km_train_data.loc[
                ~km_train_data["monthly_total_item_cat_sales_sum"].duplicated()
            ]
            .query("monthly_total_item_cat_sales_sum > 0")
        )

        range_list = []
        # Do Normalization for Clustering
        km_cluster_data["monthly_total_item_cat_sales_sum_norm"] = (
            km_cluster_data["monthly_total_item_cat_sales_sum"].apply(lambda x: x ** 0.125)
        )

        # Do Clustering
        km_cluster_data["monthly_total_item_cat_sales_heat"] = (
            KMeans(n_clusters=N_CLUSTER).fit_predict(
                km_cluster_data[["monthly_total_item_cat_sales_sum_norm"]]
            )
        )

        # Do Heat Adjustment by range - add to list
        for it in range(N_CLUSTER):
            temp_mapper = km_cluster_data.query(f"monthly_total_item_cat_sales_heat == {it}")
            # Don`t forget to denormalize
            range_tuple = (
                temp_mapper["monthly_total_item_cat_sales_sum"].min(),
                temp_mapper["monthly_total_item_cat_sales_sum"].max(),
                it + 1
            )
            range_list.append(range_tuple)

        # Join Final Result of Joined Data
        final_data = km_train_data.join(
            km_cluster_data["monthly_total_item_cat_sales_heat"],
            how='left'
        ).copy()

        # Replace Heat Value with item sales sum is zero
        final_data.loc[
            final_data["monthly_total_item_cat_sales_sum"] == 0.0,
            "monthly_total_item_cat_sales_heat"
        ] = 0

        for lb, ub, label_value in range_list:
            # print(lb, ub, label_value)

            final_data.loc[
                (final_data["monthly_total_item_cat_sales_sum"] >= lb)
              & (final_data["monthly_total_item_cat_sales_sum"] <= ub),
                "monthly_total_item_cat_sales_heat"
            ] = label_value

        # print(len(final_data))
        # print(final_data)

        # Sort List
        range_list.sort(key=lambda x: x[0])
        # print(range_list)

        # Re-mapping sales heat
        heat_mapper = {0: 0}
        for it in range(N_CLUSTER):
            heat_mapper[
                range_list[it][2]
            ] = it + 1
        final_data["monthly_total_item_cat_sales_heat"].replace(
            heat_mapper,
            inplace=True
        )

        # print(final_data)
        assert not final_data.isna().values.any(), "Final Data Contains NaN"
        final_data_list.append(final_data)

    monthly_total_item_cat_sales_with_heat = pd.concat(
        final_data_list,
        axis=0,
    ).sort_index()

    print(monthly_total_item_cat_sales_with_heat)
    print(len(monthly_total_item_cat_sales_with_heat))
    assert len(monthly_total_item_cat_sales_with_heat) == 34*84, "Length of output is incorrect"

    return monthly_total_item_cat_sales_with_heat
