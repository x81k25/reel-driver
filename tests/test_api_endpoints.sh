#!/bin/bash

# Reel Driver API Endpoint Test Script
# Tests all API endpoints with verbose output
# Usage: ./test_api_endpoints.sh [dev|stg|prod]
# Default: dev

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Environment configuration based on CLAUDE.md
ENV="${1:-dev}"

case $ENV in
    "dev")
        HOST="localhost"
        PORT="30802"
        ;;
    "stg")
        HOST="localhost"
        PORT="30801"
        ;;
    "prod")
        HOST="localhost"
        PORT="30800"
        ;;
    *)
        echo -e "${RED}ERROR: Invalid environment. Use dev, stg, or prod${NC}"
        exit 1
        ;;
esac

BASE_URL="http://${HOST}:${PORT}"
PREFIX="/reel-driver"

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Helper function to print test results
print_test_result() {
    local test_name="$1"
    local expected_code="$2"
    local actual_code="$3"
    local response="$4"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo -e "${BLUE}=== $test_name ===${NC}"
    echo -e "${YELLOW}Expected: HTTP $expected_code${NC}"
    echo -e "${YELLOW}Actual: HTTP $actual_code${NC}"
    
    if [ "$expected_code" = "$actual_code" ]; then
        echo -e "${GREEN}✓ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    echo -e "${YELLOW}Response:${NC}"
    echo "$response" | jq . 2>/dev/null || echo "$response"
    echo ""
}

# Helper function to make HTTP request and capture response
make_request() {
    local method="$1"
    local url="$2"
    local data="$3"
    local content_type="${4:-application/json}"
    
    # Create temporary files for response and HTTP code
    local temp_response=$(mktemp)
    local temp_code=$(mktemp)
    
    if [ "$method" = "GET" ]; then
        curl -s -w "%{http_code}" -o "$temp_response" "$url" > "$temp_code"
    else
        if [ -n "$data" ]; then
            curl -s -w "%{http_code}" -o "$temp_response" -X "$method" -H "Content-Type: $content_type" -d "$data" "$url" > "$temp_code"
        else
            curl -s -w "%{http_code}" -o "$temp_response" -X "$method" "$url" > "$temp_code"
        fi
    fi
    
    # Read HTTP code and response body
    local http_code=$(cat "$temp_code")
    local response_body=$(cat "$temp_response")
    
    # Clean up temporary files
    rm -f "$temp_response" "$temp_code"
    
    echo "$http_code|$response_body"
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Reel Driver API Endpoint Test Suite  ${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Environment: $ENV${NC}"
echo -e "${YELLOW}Base URL: $BASE_URL$PREFIX${NC}"
echo -e "${YELLOW}Timestamp: $(date)${NC}"
echo ""

# Test 1: Root endpoint
echo -e "${BLUE}Starting API endpoint tests...${NC}"
echo ""

result=$(make_request "GET" "$BASE_URL$PREFIX/")
http_code=$(echo "$result" | cut -d'|' -f1)
response_body=$(echo "$result" | cut -d'|' -f2-)
print_test_result "Root Endpoint" "200" "$http_code" "$response_body"

# Test 2: Health endpoint
result=$(make_request "GET" "$BASE_URL$PREFIX/health")
http_code=$(echo "$result" | cut -d'|' -f1)
response_body=$(echo "$result" | cut -d'|' -f2-)
print_test_result "Health Check" "200" "$http_code" "$response_body"

# Test 3: OpenAPI JSON
result=$(make_request "GET" "$BASE_URL$PREFIX/openapi.json")
http_code=$(echo "$result" | cut -d'|' -f1)
response_body=$(echo "$result" | cut -d'|' -f2-)
print_test_result "OpenAPI JSON" "200" "$http_code" "$response_body"

# Test 4: Root OpenAPI JSON
result=$(make_request "GET" "$BASE_URL/openapi.json")
http_code=$(echo "$result" | cut -d'|' -f1)
response_body=$(echo "$result" | cut -d'|' -f2-)
print_test_result "Root OpenAPI JSON" "200" "$http_code" "$response_body"

# Test 5: Single Prediction - Valid Input
single_prediction_data='{
    "imdb_id": "tt0111161",
    "title": "The Shawshank Redemption",
    "year": 1994,
    "rating": 9.3,
    "genre": ["Drama", "Crime"],
    "runtime": 142,
    "budget": 25000000,
    "revenue": 16000000,
    "release_year": 1994,
    "production_status": "Released",
    "original_language": "en",
    "origin_country": ["US"],
    "production_countries": ["US"],
    "spoken_languages": ["en"],
    "tmdb_rating": 8.7,
    "tmdb_votes": 26000,
    "rt_score": 91,
    "metascore": 82,
    "imdb_rating": 9.3,
    "imdb_votes": 2800000
}'

result=$(make_request "POST" "$BASE_URL$PREFIX/api/predict" "$single_prediction_data")
http_code=$(echo "$result" | cut -d'|' -f1)
response_body=$(echo "$result" | cut -d'|' -f2-)
print_test_result "Single Prediction - Valid Input" "200" "$http_code" "$response_body"

# Test 6: Single Prediction - Minimal Input
minimal_prediction_data='{
    "imdb_id": "tt0068646",
    "title": "The Godfather",
    "year": 1972,
    "rating": 9.2
}'

result=$(make_request "POST" "$BASE_URL$PREFIX/api/predict" "$minimal_prediction_data")
http_code=$(echo "$result" | cut -d'|' -f1)
response_body=$(echo "$result" | cut -d'|' -f2-)
print_test_result "Single Prediction - Minimal Input" "200" "$http_code" "$response_body"

# Test 7: Single Prediction - Invalid Input
invalid_prediction_data='{
    "imdb_id": "invalid_id",
    "title": "Test Movie",
    "year": "not_a_number",
    "rating": "invalid_rating"
}'

result=$(make_request "POST" "$BASE_URL$PREFIX/api/predict" "$invalid_prediction_data")
http_code=$(echo "$result" | cut -d'|' -f1)
response_body=$(echo "$result" | cut -d'|' -f2-)
print_test_result "Single Prediction - Invalid Input" "422" "$http_code" "$response_body"

# Test 8: Batch Prediction - Valid Input
batch_prediction_data='{
    "items": [
        {
            "imdb_id": "tt0111161",
            "title": "The Shawshank Redemption",
            "year": 1994,
            "rating": 9.3,
            "genre": ["Drama", "Crime"],
            "runtime": 142
        },
        {
            "imdb_id": "tt0068646",
            "title": "The Godfather",
            "year": 1972,
            "rating": 9.2,
            "genre": ["Crime", "Drama"],
            "runtime": 175
        }
    ]
}'

result=$(make_request "POST" "$BASE_URL$PREFIX/api/predict_batch" "$batch_prediction_data")
http_code=$(echo "$result" | cut -d'|' -f1)
response_body=$(echo "$result" | cut -d'|' -f2-)
print_test_result "Batch Prediction - Valid Input" "200" "$http_code" "$response_body"

# Test 9: Batch Prediction - Empty List
empty_batch_data='{
    "items": []
}'

result=$(make_request "POST" "$BASE_URL$PREFIX/api/predict_batch" "$empty_batch_data")
http_code=$(echo "$result" | cut -d'|' -f1)
response_body=$(echo "$result" | cut -d'|' -f2-)
print_test_result "Batch Prediction - Empty List" "200" "$http_code" "$response_body"

# Test 10: Batch Prediction - Invalid Format
invalid_batch_data='[
    {
        "imdb_id": "tt0111161",
        "title": "The Shawshank Redemption",
        "year": 1994,
        "rating": 9.3
    }
]'

result=$(make_request "POST" "$BASE_URL$PREFIX/api/predict_batch" "$invalid_batch_data")
http_code=$(echo "$result" | cut -d'|' -f1)
response_body=$(echo "$result" | cut -d'|' -f2-)
print_test_result "Batch Prediction - Invalid Format" "422" "$http_code" "$response_body"

# Test 11: 404 Error - Non-existent endpoint
result=$(make_request "GET" "$BASE_URL$PREFIX/nonexistent")
http_code=$(echo "$result" | cut -d'|' -f1)
response_body=$(echo "$result" | cut -d'|' -f2-)
print_test_result "404 Error - Non-existent Endpoint" "404" "$http_code" "$response_body"

# Test 12: Method Not Allowed
result=$(make_request "DELETE" "$BASE_URL$PREFIX/api/predict")
http_code=$(echo "$result" | cut -d'|' -f1)
response_body=$(echo "$result" | cut -d'|' -f2-)
print_test_result "Method Not Allowed - DELETE on Predict" "405" "$http_code" "$response_body"

# Test 13: Malformed JSON
malformed_json='{"imdb_id": "tt0111161", "title": "Test", "year": 1994, "rating": }'

result=$(make_request "POST" "$BASE_URL$PREFIX/api/predict" "$malformed_json")
http_code=$(echo "$result" | cut -d'|' -f1)
response_body=$(echo "$result" | cut -d'|' -f2-)
print_test_result "Malformed JSON Request" "422" "$http_code" "$response_body"

# Test 14: Health endpoint without prefix (should fail)
result=$(make_request "GET" "$BASE_URL/health")
http_code=$(echo "$result" | cut -d'|' -f1)
response_body=$(echo "$result" | cut -d'|' -f2-)
print_test_result "Health Without Prefix (Expected Fail)" "404" "$http_code" "$response_body"

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}           Test Summary                 ${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Environment: $ENV${NC}"
echo -e "${YELLOW}Total Tests: $TOTAL_TESTS${NC}"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Check the output above for details.${NC}"
    exit 1
fi