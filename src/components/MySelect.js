import { Select } from 'antd';

const { Option } = Select;

function MySelect({ options, value, onChange }) {
    const handleSelectChange = value => {
        console.log('选择的值为', value);
        const label = options.find(option => option.value === value)?.label;
        console.log('选择的标签为', label);
        if (onChange) {
            onChange(value);
        }
    };

    return (
        <Select
            value={value}
            onChange={handleSelectChange}
            style={{
                // width: 80,
                borderBottom: '1px solid black',
                width: '40%',
                height: '25px',
                overflow: 'hidden',
            }}
        >
            {options.map(option => (
                <Option key={option.value} value={option.value}>
                    {option.label}
                </Option>
            ))}
        </Select>
    );
}

export default MySelect;
