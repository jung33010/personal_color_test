-- 1. 데이터베이스(Schema) 생성 (이미 존재하면 무시)
CREATE DATABASE IF NOT EXISTS personal_color_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 2. 해당 데이터베이스 사용
USE personal_color_db;

-- 4. UsageLog 최신 테이블 생성
CREATE TABLE UsageLog (
    id INT AUTO_INCREMENT PRIMARY KEY,            -- 자동 증가하는 고유 ID
    session_uuid VARCHAR(36),                     -- 브라우저 세션 고유 ID
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, -- 데이터 입력 시간
    lighting_env VARCHAR(50),                     -- 조명 환경 (예: 실내/실외)
    ai_result TEXT,                               -- AI가 분석한 진단 결과
    is_completed BOOLEAN DEFAULT FALSE,           -- 성공 여부 (0: 실패, 1: 성공)
    prompt_tokens INT DEFAULT 0,                  -- 입력 토큰 수
    completion_tokens INT DEFAULT 0,              -- 출력 토큰 수
    total_cost FLOAT DEFAULT 0.0,                 -- 소모된 총 비용 (USD)
    image_path VARCHAR(255),                      -- 업로드 이미지 저장 경로
    processing_time FLOAT,                        -- 응답 소요 시간 (초)
    user_feedback VARCHAR(50),                    -- 사용자 만족도 평가
    device_type VARCHAR(50),                      -- 접속 기기 종류 (PC/모바일)
    retry_count INT DEFAULT 0,                    -- 해당 세션 내 재시도 횟수
    model_version VARCHAR(50)                     -- 사용된 AI 모델 버전
);
