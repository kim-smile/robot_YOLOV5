import serial

def execute_motion(port, motion_id, parent=None):
    # 모션 제어 기본 패킷 생성
    packet_buff = [0xff, 0xff, 0x4c, 0x53,  # 헤더
                   0x00, 0x00, 0x00, 0x00,  # Destination ADD, Source ADD
                   0x30, 0x0c, 0x03,        # 0x30 실행 명령어 0x0c 모션실행 0x03 파라메타 길이
                   motion_id, 0x00, 0x64,   # 모션 ID, 모션 반복, 모션 속도 지정
                   0x00]                    # 체크섬

    # 체크섬 계산
    checksum = 0
    for i in range(6, 14):
        checksum += packet_buff[i]
    packet_buff[14] = checksum

    # 시리얼 포트 열기
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        if ser.is_open:
            # 패킷 전송
            ser.write(packet_buff)
        else:
            raise serial.SerialException(f"Failed to open serial port {port}.")
    except serial.SerialException as e:
        # 경고 메시지 박스 표시
        if parent:
            print(f"Serial Port Error", str(e))
        else:
            print(f"Serial Port Error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()