#!/bin/bash

# Configuration variables
IMAGE_NAME="cn-lexi-mail-ai"
IMAGE_VERSION="1.0.0"
USERNAME="veer034"  # Your GitHub username
REGISTRY="ghcr.io"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print banner
echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}  Email Classifier Service Docker Build & Push Tool    ${NC}"
echo -e "${GREEN}======================================================${NC}"

# Full image name with registry
FULL_IMAGE_NAME="$REGISTRY/$USERNAME/$IMAGE_NAME:$IMAGE_VERSION"

# Prompt for the GitHub Personal Access Token (PAT)
read -sp "Enter your GitHub Personal Access Token: " PAT
echo ""

# Function to check if image exists in registry
check_image_exists() {
    echo -e "${YELLOW}Checking if $IMAGE_NAME:$IMAGE_VERSION already exists in the registry...${NC}"
    
    # Get token for API access
    TOKEN=$(curl -s -u "$USERNAME:$PAT" \
        "https://ghcr.io/token?scope=repository:$USERNAME/$IMAGE_NAME:pull" \
        | grep -o '"token":"[^"]*' | cut -d':' -f2 | tr -d '"')
    
    if [ -z "$TOKEN" ]; then
        echo -e "${RED}Failed to get authentication token. Check your credentials.${NC}"
        exit 1
    fi
    
    # Check if image exists
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer $TOKEN" \
        "https://ghcr.io/v2/$USERNAME/$IMAGE_NAME/manifests/$IMAGE_VERSION")
    
    if [ "$HTTP_CODE" == "200" ]; then
        echo -e "${YELLOW}Image $FULL_IMAGE_NAME already exists in the registry.${NC}"
        read -p "Do you want to overwrite it? (y/n): " OVERWRITE
        if [ "$OVERWRITE" != "y" ]; then
            echo -e "${YELLOW}Aborting.${NC}"
            exit 0
        fi
    fi
}

# Build the Docker image
build_image() {
    echo -e "${YELLOW}Building Docker image $IMAGE_NAME:$IMAGE_VERSION...${NC}"
    
    if docker build -t "$IMAGE_NAME:$IMAGE_VERSION" .; then
        echo -e "${GREEN}Docker image built successfully.${NC}"
    else
        echo -e "${RED}Failed to build Docker image.${NC}"
        exit 1
    fi
}

# Tag the image for GitHub Container Registry
tag_image() {
    echo -e "${YELLOW}Tagging image as $FULL_IMAGE_NAME...${NC}"
    
    if docker tag "$IMAGE_NAME:$IMAGE_VERSION" "$FULL_IMAGE_NAME"; then
        echo -e "${GREEN}Image tagged successfully.${NC}"
    else
        echo -e "${RED}Failed to tag image.${NC}"
        exit 1
    fi
}

# Login to GitHub Container Registry
login_to_registry() {
    echo -e "${YELLOW}Logging into GitHub Container Registry...${NC}"
    
    if echo "$PAT" | docker login "$REGISTRY" -u "$USERNAME" --password-stdin; then
        echo -e "${GREEN}Login successful.${NC}"
    else
        echo -e "${RED}Login failed. Please check your credentials.${NC}"
        exit 1
    fi
}

# Push the image to GitHub Container Registry
push_image() {
    echo -e "${YELLOW}Pushing image to GitHub Container Registry...${NC}"
    
    if docker push "$FULL_IMAGE_NAME"; then
        echo -e "${GREEN}Image pushed successfully.${NC}"
    else
        echo -e "${RED}Failed to push image.${NC}"
        exit 1
    fi
}

# Main execution
check_image_exists
build_image
tag_image
login_to_registry
push_image

echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}  Email Classifier Service image build and push complete!${NC}"
echo -e "${GREEN}  Image: $FULL_IMAGE_NAME${NC}"
echo -e "${GREEN}======================================================${NC}"

# Cleanup (optional)
read -p "Do you want to remove the local Docker image to free up space? (y/n): " CLEANUP
if [ "$CLEANUP" == "y" ]; then
    docker rmi "$FULL_IMAGE_NAME" "$IMAGE_NAME:$IMAGE_VERSION"
    echo -e "${GREEN}Local images removed.${NC}"
fi