"""
Utility script to create QR codes for customers.
Run this script to generate QR codes that customers can use.

Usage:
    python create_qr_code.py --customer-id CUST_001 --name "Nguyễn Văn A" --phone "0123456789"
    
Or interactive mode:
    python create_qr_code.py
"""

import argparse
import json
import os
import qrcode
from PIL import Image, ImageDraw, ImageFont

def create_customer_qr(customer_id, name=None, phone=None, output_dir="data/qr_codes"):
    """
    Create QR code for a customer.
    
    Args:
        customer_id: Customer ID (e.g., "CUST_001")
        name: Customer name (optional)
        phone: Customer phone (optional)
        output_dir: Directory to save QR codes
    
    Returns:
        Path to saved QR code image
    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create QR data (JSON format)
    qr_data = {
        "customer_id": customer_id
    }
    
    if name:
        qr_data["name"] = name
    if phone:
        qr_data["phone"] = phone
    
    # Convert to JSON string
    qr_text = json.dumps(qr_data, ensure_ascii=False)
    
    # Also support simple text format (for backward compatibility)
    # Uncomment if you want simple text instead of JSON:
    # qr_text = customer_id
    
    # Create QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(qr_text)
    qr.make(fit=True)
    
    # Create image
    qr_img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to RGB if needed (PIL requires RGB for some operations)
    if qr_img.mode != 'RGB':
        qr_img = qr_img.convert('RGB')
    
    # Simple version: Save QR code directly (without text label)
    # If you want text label, uncomment the code below
    filename = os.path.join(output_dir, f"{customer_id}.png")
    qr_img.save(filename)
    
    # Optional: Add text label below QR code
    # Uncomment this section if you want text labels
    """
    # Get QR code dimensions
    qr_width, qr_height = qr_img.size
    
    # Prepare text lines
    text_lines = [f"ID: {customer_id}"]
    if name:
        text_lines.append(name)
    
    # Get font for text
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
    
    # Create a temporary image to measure text
    temp_img = Image.new('RGB', (100, 100), 'white')
    draw_temp = ImageDraw.Draw(temp_img)
    
    # Calculate text dimensions
    line_height = 25
    max_text_width = 0
    for line in text_lines:
        bbox = draw_temp.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        max_text_width = max(max_text_width, text_width)
    
    # Calculate total image size (QR + text area)
    text_area_height = len(text_lines) * line_height + 20
    total_height = qr_height + text_area_height
    total_width = max(qr_width, max_text_width + 20)
    
    # Create new image with space for text
    img_with_label = Image.new('RGB', (total_width, total_height), 'white')
    
    # Paste QR code at top (use box format: (left, top, right, bottom))
    qr_x = (total_width - qr_width) // 2
    paste_box = (qr_x, 0, qr_x + qr_width, qr_height)
    img_with_label.paste(qr_img, paste_box)
    
    # Draw customer ID text
    draw = ImageDraw.Draw(img_with_label)
    
    # Draw text lines (centered)
    text_y = qr_height + 10
    for i, line in enumerate(text_lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (total_width - text_width) // 2
        draw.text((text_x, text_y + i * line_height), line, fill="black", font=font)
    
    # Save file with label
    img_with_label.save(filename)
    """
    
    print(f"✅ Created QR code: {filename}")
    print(f"   Customer ID: {customer_id}")
    if name:
        print(f"   Name: {name}")
    if phone:
        print(f"   Phone: {phone}")
    print(f"   QR Data: {qr_text}")
    print()
    
    return filename

def create_multiple_qr_codes(start_id=1, count=10, prefix="CUST_", output_dir="data/qr_codes"):
    """Create multiple QR codes at once."""
    print(f"Creating {count} QR codes...")
    print()
    
    for i in range(start_id, start_id + count):
        customer_id = f"{prefix}{i:03d}"  # Format: CUST_001, CUST_002, ...
        create_customer_qr(customer_id, output_dir=output_dir)
    
    print(f"✅ Created {count} QR codes in '{output_dir}' directory")

def main():
    parser = argparse.ArgumentParser(description='Create QR codes for customers')
    parser.add_argument('--customer-id', '-c', type=str, help='Customer ID (e.g., CUST_001)')
    parser.add_argument('--name', '-n', type=str, help='Customer name')
    parser.add_argument('--phone', '-p', type=str, help='Customer phone')
    parser.add_argument('--batch', '-b', type=int, help='Create multiple QR codes (count)')
    parser.add_argument('--start-id', '-s', type=int, default=1, help='Starting ID for batch mode')
    parser.add_argument('--prefix', type=str, default='CUST_', help='Prefix for customer ID')
    
    args = parser.parse_args()
    
    # Interactive mode if no arguments
    if not args.customer_id and not args.batch:
        print("=" * 60)
        print("QR Code Generator for Customers")
        print("=" * 60)
        print()
        
        customer_id = input("Enter Customer ID (e.g., CUST_001): ").strip()
        if not customer_id:
            print("❌ Customer ID is required!")
            return
        
        name = input("Enter Customer Name (optional): ").strip() or None
        phone = input("Enter Customer Phone (optional): ").strip() or None
        
        create_customer_qr(customer_id, name, phone)
        return
    
    # Batch mode
    if args.batch:
        create_multiple_qr_codes(args.start_id, args.batch, args.prefix, output_dir="data/qr_codes")
        return
    
    # Single QR code mode
    if args.customer_id:
        create_customer_qr(args.customer_id, args.name, args.phone)
    else:
        parser.print_help()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

