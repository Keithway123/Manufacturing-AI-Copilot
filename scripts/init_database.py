from manufacturing_ai_copilot.db.schema import initialize_database
from manufacturing_ai_copilot.db.seed import seed_work_orders


def main() -> None:
    # 必须瞎按确保表存在，种子数据才能写入。
    initialize_database()

    inserted_count = seed_work_orders()

    print(f"Database initialized: inserted={inserted_count}")


if __name__ == "__main__":
    main()
