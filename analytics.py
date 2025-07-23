import csv
import time
from collections import deque, Counter
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

DWELL_THRESHOLD = 2.0

class RetailMetrics:
    def __init__(self):
        self.face_records = {} # id -> deque[(timestamp, expression, age, gender)]

    def update(self, obj_id: int, expression: str, age: int, gender: str):
        if obj_id not in self.face_records:
            self.face_records[obj_id] = deque(maxlen=150) # 約5秒分(30fps時)のデータを保持
        
        self.face_records[obj_id].append((time.time(), expression, age, gender))

    def get_person_summary(self, obj_id: int) -> Dict[str, Any]:
        records = self.face_records.get(obj_id)
        if not records:
            return {}

        # データの抽出
        timestamps, expressions, ages, genders = zip(*records)

        # 滞在時間の計算
        dwell_sec = timestamps[-1] - timestamps[0]
        
        # 結果の判定
        result = 'stay' if dwell_sec >= DWELL_THRESHOLD else 'pass'
        
        # 最も頻繁に現れた表情と性別
        top_expression = Counter(expressions).most_common(1)[0][0]
        top_gender = Counter(genders).most_common(1)[0][0]

        # 年齢の中央値（安定化のため）
        stable_age = int(np.median(ages))

        return {
            "gender": top_gender,
            "age": stable_age,
            "expression": top_expression,
            "result": result,
            "dwell_sec": dwell_sec
        }
    
    def get_current_stable_attributes(self, obj_id: int) -> Dict[str, Any]:
        """画面表示用に、直近データから安定した属性を返す"""
        records = self.face_records.get(obj_id)
        if not records:
            return {"age": "?", "gender": "?", "expression": "?"}
        
        # 直近5フレームのデータで計算
        recent_records = list(records)[-5:]
        _, expressions, ages, genders = zip(*recent_records)

        return {
            "age": int(np.median(ages)),
            "gender": Counter(genders).most_common(1)[0][0],
            "expression": Counter(expressions).most_common(1)[0][0]
        }


    def finalize_person(self, obj_id: int):
        if obj_id in self.face_records:
            del self.face_records[obj_id]

class CsvLogger:
    def __init__(self, out_dir='logs'):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M')
        self.path = Path(out_dir) / f'analytics_{ts}.csv'
        self.file = open(self.path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            ['end_timestamp', 'gender', 'age_stable', 'top_expression', 'result', 'total_dwell_sec'])

    def log(self, summary: Dict):
        self.writer.writerow([
            time.strftime('%Y-%m-%d %H:%M:%S'),
            summary.get("gender", "N/A"),
            summary.get("age", "N/A"),
            summary.get("expression", "N/A"),
            summary.get("result", "N/A"),
            f'{summary.get("dwell_sec", 0):.2f}'
        ])

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
